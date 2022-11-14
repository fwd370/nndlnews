# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F

import re
import pandas as pd
import nltk
import contractions

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

'''
HW 6
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
'''

MAX_LENGTH = 300

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<SOS>",1:"<EOS>",2:"<UNK>",3:"<PAD>"}
        self.n_words = 4 #Start count with "SOS", "EOS", UNK, PAD
    
    def addToVocab(self, sentence: str):
        for word in sentence.split(" "):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words+= 1
                self.word2count[word] = 1
            else:
                self.word2count[word] += 1
    
    def convertSentenceToIndex(self, sentence: str):
        idxs = [self.word2index[word] for word in sentence.split(" ")]
        idxs.append(EOS_token)
        #return torch.tensor(idxs, dtype=torch.long, device=device).view(-1,1)
    
    def makeDataTensor(self, sentPair: tuple):
        inputTensor = self.convertSentenceToIndex(sentPair[0])
        targetTensor = self.convertSentenceToIndex(sentPair[1])
        return (inputTensor,targetTensor)

def processString(text: str):
    text = re.sub("\\\\n","",text)                  # Remove Line breaks in the article
    text = re.sub("SINGAPORE","",text)              # Unique processing for Today articles
    text = contractions.fix(text)                   # Replaces apostrophes with full word to reduce dimensionality
    text = re.sub(r"([.!?])",r" \1", text)
    text = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", text)
    text = text.lower().strip()
    text = re.sub(r"[0-9]+","N",text)

    return text

def dataTo():
    pass


strText='[\'SINGAPORE — The Health Sciences Authority (HSA) has authorised the Pfizer Comirnaty bivalent Covid-19 vaccine for use in Singapore, as a booster vaccine for those aged 12 and above who have received their primary series vaccination.\\n\', \'The Ministry of Health (MOH) earlier this month rolled out the Moderna Spikevax bivalent vaccine, which was also authorised for use by HSA.\\n\', \'In a press release on Tuesday (Oct 25), HSA said that the Pfizer Comirnaty vaccine was granted interim authorisation on Oct 11 under the Pandemic Special Access Route. This facilitates early access to critical novel vaccines and medicine during a pandemic.\\n\', \'The latest vaccine is an updated version of the original Pfizer Comirnaty vaccine.\\n\', \'It comprises two messenger ribonucleic acid (mRNA) components — 15 micrograms targeting the original Sars-CoV-2 coronavirus, as well as 15 micrograms targeting the Omicron BA.4 and BA.5 virus strains.\\n\', \'HSA said that the authorisation is based on the totality of available evidence, including Pfizer’s non-clinical studies, clinical trials with different variant-updated vaccines, and the quality and manufacturing processes.\\n\', \'"Based on the available information, HSA has assessed that the benefits are expected to outweigh the risks for use of Comirnaty bivalent vaccine as a booster to protect against Covid-19 as the virus continues to evolve."\\n\', \'HSA added that MOH and the expert committee on Covid-19 vaccination will issue the official vaccination recommendations on its usage when ready.\\n\', "As the clinical study by Pfizer for its bivalent vaccine is still under way, HSA said that its assessment of the vaccine\'s efficacy was based primarily on an earlier clinical study conducted on people aged above 55.\\n", \'HSA added that it considers such data as relevant since these Omicron subvariants are closely related.\\n\', \'The study showed that the bivalent vaccine elicited a stronger immune response against the targeted Omicron BA.1 strain, while still maintaining an adequate response against the original Sars-CoV-2 virus.\\n\', \'"This indicates that bivalent vaccines provide a broader immunity and better protection against the Sars-CoV-2 virus."\\n\', \'Another ongoing clinical study on the vaccine involving 80 participants aged 18 and above also showed consistent trends of higher immune responses against the Omicron BA.4 and BA.5 subvariants.\\n\', \'Based on the overall data, HSA said that there is a sufficient body of evidence to support the safe use of the Comirnaty bivalent vaccine.\\n\', \'Thus, it will extend its use to adults and adolescents aged above 12 years based on the following:\\n\', \'The younger population generally develops a higher immune response compared to the older population due to the gradual weakening of the immune system as a person ages. So, the bivalent vaccine would trigger "comparable or better immune response in adolescents than older adults"\\n\', "There is a large amount of real-world data from Singapore and overseas surveillance studies supporting the original Comirnaty vaccine\'s effectiveness and safety. It is reasonable to conclude that the variant-updated bivalent vaccines will retain similar effectiveness and safety profiles\\n", \'Safety data from the clinical studies also showed that the bivalent vaccine was generally well-tolerated, HSA said.\\n\', \'The adverse events or reactions to the vaccines were mostly mild to moderate, such as injection site pain or tenderness, or both, as well as fatigue, headache and muscle pain.\\n\', \'"These reactions are generally associated with vaccinations and expected as part of the body’s natural response to build immunity against Covid-19. They usually resolve on their own within a few days," HSA added.\\n\', \'The authority said that it will continue to actively monitor the safety of the vaccine and require Pfizer to submit data from the ongoing clinical study for the bivalent vaccine, to ensure that the benefits continue to outweigh the risks.\\n\', \'"HSA will take the necessary actions and provide updates to the public if significant safety concerns are identified."\\n\']'
print(processString(strText))
