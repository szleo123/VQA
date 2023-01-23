import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from variables import *
from clipvqa import CLIPVQADataset, VQA, VQAEval
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import os
import json
import clip

class ImageEncoder(nn.Module):
    def __init__(self, model):
        super(ImageEncoder, self).__init__()
        self.model = model
        self.fc = nn.Linear(512, 512)

    def forward(self, image):
        with torch.no_grad():
            img_feature = self.model.encode_image(image)
        img_feature = self.fc(img_feature.to(torch.float))
        return img_feature

class TextEncoder(nn.Module):
    def __init__(self, model):
        super(TextEncoder, self).__init__()
        self.model = model
        self.fc = nn.Linear(512, 512)

    def forward(self, question):
        with torch.no_grad():
            question = self.model.encode_text(question).to(torch.float)
        question = self.fc(question)
        return question

class Model(nn.Module):
    def __init__(self, ans_vocab_size):
        super(Model, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device="cuda:0")
        self.image_encoder = ImageEncoder(self.model)
        self.text_encoder = TextEncoder(self.model)
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, ans_vocab_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(ans_vocab_size, ans_vocab_size))

    def forward(self, image, question):
        combined = self.image_encoder(image) * self.text_encoder(question)
        return self.mlp(combined)
    
def evaluate(name):

    # set up file names and paths
    annFile     ='../../data/val/val_annotations.json'
    quesFile    ='../../data/val/val_questions.json'

    # An example result json file has been provided in './Results' folder.  

    resFile = '../../output/%s.json'%name

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate() 
    with open('../../ouptut/'+name+".txt",'w') as f:

        # print accuracies
        f.write("\n")
        f.write("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
        f.write("Per Question Type Accuracy is the following:")
        f.write("\n")
        for quesType in dict(sorted(vqaEval.accuracy['perQuestionType'].items(), key = lambda x: -x[1])):
            f.write("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            f.write("\n")
        f.write("\n")
        f.write("Per Answer Type Accuracy is the following:")
        f.write("\n")
        for ansType in vqaEval.accuracy['perAnswerType']:
            f.write("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            f.write("\n")
        f.write("\n")

if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    trainLoader = DataLoader(
            dataset=CLIPVQADataset(
            input_file='train.npy'),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKER)
    valLoader = DataLoader(
            dataset=CLIPVQADataset(
            input_file='val.npy'),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKER)
    model = Model(trainLoader.dataset.getSize("answer")).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    for epoch in range(EPOCH):
        trainLoss=0
        valLoss=0

        model.train()
        for idx, (img, questions, answer_ids, _) in enumerate(tqdm(trainLoader)):
            image = img.to(device=DEVICE)
            questions = clip.tokenize(questions).to(device=DEVICE)
            answers = answer_ids.type(torch.LongTensor).to(device=DEVICE)

            predictions = model(image, questions)
            loss = criterion(predictions, answers)
            trainLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        trainLoss /= len(trainLoader.dataset)

        model.eval()
        output=[]
        for idx, (img, questions, answer_ids, question_ids) in enumerate(tqdm(valLoader)):
            image = img.to(device=DEVICE)
            questions = clip.tokenize(questions).to(device=DEVICE)
            answers = answer_ids.type(torch.LongTensor).to(device=DEVICE)

            with torch.no_grad():
                predictions = model(image, questions)
                loss = criterion(predictions, answers)

            valLoss += loss.item()
            prediction_ids = predictions.argmax(-1)
            for question, prediction in zip(question_ids, prediction_ids):
                output.append({'question_id': question.item(), 'answer': valLoader.dataset.answer_corpus.idx2word(prediction)})
        valLoss /= len(valLoader.dataset)

        saveName = 'clip' + str(epoch)
        with open('../../outpu/' + saveName + '.json', 'w') as out:
            json.dump(output, out)
            
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, saveName+'.pth'))

        print('Epoch:{}/{} | Training Loss: {} | Validation Loss: {}'.format(epoch+1, EPOCH, trainLoss, valLoss))

        evaluate(saveName)
        