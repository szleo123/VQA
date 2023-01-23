import os
from PIL import Image
from variables import *
import re
import json
from collections import defaultdict
import numpy as np

def resizeImage():
    if not os.path.exists(RESIZE_PATH):
        os.makedirs(RESIZE_PATH)
    for name, imagePath in {"train2014": TRAIN_IMAGE_PATH, "val2014": VAL_IMAGE_PATH}.items():
        if not os.path.exists(RESIZE_PATH+'/'+name):
            os.makedirs(RESIZE_PATH+'/'+name)
        images = os.listdir(imagePath)
        count=0
        print(len(images))
        for image in images:
            count+=1
            with open(os.path.join(imagePath, image), 'r+b') as f:
                with Image.open(f) as img:
                    if count % 100 == 0:
                        print(count)
                    img = img.resize([IMAGE_SIZE, IMAGE_SIZE], Image.ANTIALIAS)
                    img.save(os.path.join(RESIZE_PATH+'/'+name, image), img.format)

def createCorpus():
    regex = re.compile(r'(\W+)')
    q_vocab = []
    for questionPath in [TRAIN_QUESTION_PATH, VAL_QUESTION_PATH]:
        with open(questionPath, 'r') as f:
            data = json.load(f)
        questions = data['questions']
        for question in questions:
            split = regex.split(question['question'].lower())
            tmp = [w.strip() for w in split if len(w.strip()) > 0]
            q_vocab.extend(tmp)

    q_vocab = list(set(q_vocab))
    q_vocab.sort()
    q_vocab = ['<pad>'] + ['<unk>'] + q_vocab
    
    if not os.path.exists(DATA_PATH): 
        os.makedirs(DATA_PATH)
    with open(DATA_PATH + '/question_vocabs.txt', 'w') as f:
        f.writelines([v+'\n' for v in q_vocab])

    answers = defaultdict(int)
    for annotationPath in [TRAIN_ANNOTATION_PATH, VAL_ANNOTATION_PATH]:
        with open(annotationPath, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        for annotation in annotations:
            for ans in annotation['answers']:
                vocab = ans['answer']
                if not re.search(r'[^\w\s]', vocab):
                    answers[vocab] += 1

    answers = sorted(answers, key=answers.get, reverse= True)
    top_answers = ['<unk>'] + answers[:TOP-1]
    with open(DATA_PATH + '/annotation_vocabs.txt', 'w') as f :
        f.writelines([ans+'\n' for ans in top_answers])

def markImage(type):
    def tokenizer(sentence):
        regex = re.compile(r'(\W+)')
        tokens = regex.split(sentence.lower())
        tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
        return tokens

    def match_top_ans(annotation_ans):
        annotation_dir = DATA_PATH + '/annotation_vocabs.txt'
        if "top_ans" not in match_top_ans.__dict__:
            with open(annotation_dir, 'r') as f:
                match_top_ans.top_ans = {line.strip() for line in f}
        annotation_ans = {ans['answer'] for ans in annotation_ans}
        valid_ans = match_top_ans.top_ans & annotation_ans

        if len(valid_ans) == 0:
            valid_ans = ['<unk>']
            match_top_ans.unk_ans += 1

        return annotation_ans, valid_ans
    questionPath = TRAIN_QUESTION_PATH
    annotationPath = TRAIN_ANNOTATION_PATH
    if type == "val2014":
        questionPath = VAL_QUESTION_PATH
        annotationPath = VAL_ANNOTATION_PATH
    with open(questionPath, 'r') as f:
        data = json.load(f)
        questions = data['questions']
        filename = data['data_subtype']
    with open(annotationPath, 'r') as f:
        annotations = json.load(f)['annotations']
    question_dict = {ans['question_id']: ans for ans in annotations}

    match_top_ans.unk_ans = 0
    dataset = [None]*len(questions)
    for idx, qu in enumerate(questions):
        qu_id = qu['question_id']
        qu_sentence = qu['question']
        qu_tokens = tokenizer(qu_sentence)
        img_id = qu['image_id']
        img_name = 'COCO_' + type + '_{:0>12d}.jpg'.format(img_id)
        img_path = os.path.join(RESIZE_PATH, filename, img_name)

        info = {'img_name': img_name,
                'img_path': img_path,
                'qu_sentence': qu_sentence,
                'qu_tokens': qu_tokens,
                'qu_id': qu_id}
        annotation_ans = question_dict[qu_id]['answers']
        all_ans, valid_ans = match_top_ans(annotation_ans)
        info['all_ans'] = all_ans
        info['valid_ans'] = valid_ans

        dataset[idx] = info

    np.save(os.path.join(DATA_PATH, f'{type}.npy'), np.array(dataset))

def createImageDataset():
    for type in ["train2014", "val2014"]:
        markImage(type)

if __name__ == "__main__":
    resizeImage()
    print("finished image resizing")
    createCorpus()
    print("finished corpus creating")
    createImageDataset()
    print("finished dataset creating")
