# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:59:31 2020

@author: aspin.c
"""
#importing libraries
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

#initializing models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#defining Ques & text
text = "My Name is ASPIN C. Am studying MSc Data Analytics"
question=[ "What is my name?","Which course i studying?"]

#testing
for i in range(2):    
    input_ids = tokenizer.encode(question[i], text)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    answer=answer.replace(' ##', '')
    print(answer)