from vqa import VQA, VQAEval
from collections import Counter
import json

annTrainFile     ='../../data/train/train_annotations.json'
quesTrainFile    ='../../data/train/train_questions.json'
annValFile     ='../../data/val/val_annotations.json'
quesValFile    ='../../data/val/val_questions.json'

def evaluate(name):

    # An example result json file has been provided in './Results' folder.  

    resFile = '../../output/%s.json'%name

    # create vqa object and vqaRes object
    vqa = VQA(annValFile, quesValFile)
    vqaRes = vqa.loadRes(resFile, quesValFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate() 
    with open('../../output/'+name+".txt",'w') as f:

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
    # initialize VQA api for QA annotations
    vqaTrain = VQA(annTrainFile, quesTrainFile)
    vqaVal = VQA(annValFile, quesValFile)

    # load question types
    with open("mscoco_question_types.txt") as f:
        question_types = []
        lines = f.readlines()
        for line in lines:
            question_types.append(line.strip())

    # find most popular answers
    majority = dict()
    for t in question_types:
        annIds = vqaTrain.getQuesIds(quesTypes=t)
        anns = vqaTrain.loadQA(annIds)
        ans = vqaTrain.getTruth(anns)
        freq = Counter()
        for a in ans:
            freq[a] += 1
        majority[t] = max(freq, key=freq.get)

    # output
    output = []
    qas = vqaVal.dataset['annotations']
    for qa in qas:
        output.append({'question_id': qa['question_id'], 'answer': majority[qa['question_type']]})
    
    saveName = "simple-baseline"
    with open('../../output/' + saveName + '.json', 'w') as out:
        json.dump(output, out)

    evaluate(saveName)