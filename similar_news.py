from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import gensim

df = pd.read_excel('C:/Users/ohee/workspace_data/date_18_5.xlsx',index=False)

doc = df['Contents']
sentences = []
for i in range(len(df['Contents'])):
    sentences.append(df['Contents'][i])
#print(sentences)

sentences = [s.lower().strip().split(" ") for s in sentences]

sim_final_over2 = []
same = []
final = []
final_real = []

tagged_documents = []

for i, s in enumerate(sentences):
    tagged_documents.append(gensim.models.doc2vec.TaggedDocument(s, [i]))

Doc2Vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1)
Doc2Vec_model.build_vocab(tagged_documents)
Doc2Vec_model.train(tagged_documents, total_examples=len(tagged_documents), epochs=300)

def calculate_sim(a):
    std_doc = word_tokenize(df['Contents'][a])
#     print("<",a, "번째 기사와의 유사도 검증결과>")
    sim_real = []
    sim_final = []
    sim_odd = []
    sim_final_final = []

    sim =Doc2Vec_model.docvecs.most_similar(positive=[Doc2Vec_model.infer_vector(std_doc)],topn=len(df['Contents']))

    #유사도 0.7 넘는 거만 sim_real에 저장
    for i in range(len(df['Contents'])):
        if(sim[i][1] >= 0.7):
            sim_real += sim[i]

    #유사도 0.7 넘는거가 있는 기사중에 제일 유사도 높은 기사가 자기자신
    #sim_final에 저장
    if(len(sim_real) != 0):
        if(sim_real[0] == a):
            sim_final = sim_real

    #sim_final이 0이 아닐(0.7 넘는 유사도 중에서 자기자신이 유사도 1등인거) 때 유사도만 뽑은 리스트 만듦
    if(len(sim_final) != 0):
        sim_odd = sim_final[1::2]
        #유사도끼리의 차를 구함
        sim_odd_minus = [sim_odd[0]- j for j in sim_odd]

        #유사도 차이가 0.02보다 작으면 그때 인덱스를 final final에 추가하면 되나
        for i in range(len(sim_odd_minus)):
            if(sim_odd_minus[i] <= 0.02):
                sim_final_final.append(sim_final[2*i])
                sim_final_final.append(sim_final[2*i+1])
    else:
        sim_final_final = sim_final

    if(len(sim_final_final) > 2):
        sim_final_over2.append(sim_final_final[0::2])


for i in range(len(df['Contents'])):
    calculate_sim(i)

#같은거 다모음
for i in range(len(sim_final_over2)):
    for j in range(len(sim_final_over2[i])):
         if(i < len(sim_final_over2)-1):
            for k in range(i+1, len(sim_final_over2)):
                 for l in range(len(sim_final_over2[k])):
                        if(sim_final_over2[i][j] == sim_final_over2[k][l]):
                            same.append(list(set(sim_final_over2[i] + sim_final_over2[k])))


for i in range(len(same)):
    for j in range(len(same[i])):
         if(i < len(same)-1):
            for k in range(i+1, len(same)):
                 for l in range(len(same[k])):
                        if(same[i][j] == same[k][l]):
                            final.append(list(set(same[i] + same[k])))
            break


for i in range(len(sim_final_over2)):
    for j in range(len(sim_final_over2[i])):
        for k in range(len(final)):
            for l in range(len(final[k])):
                if(sim_final_over2[i][j] == final[k][l]):
                    sim_final_over2[i] = final[k]


for i in range(len(sim_final_over2)-1,-1,-1):
    for j in range(len(final)):
        if(sim_final_over2[i] == final[j]):
            sim_final_over2.pop(i)

for i in final:
    for j in sim_final_over2:
        if(i == j):
            sim_final_over2.remove(j)
# #중복 of 중복
# print(final)

# #중복기사 모음집 - 중복 of 중복
# print(sim_final_over2)

sim_final_over2 = sim_final_over2 + final

# #ㄹㅇ 최종 중복리스트
# print(sim_final_over2)

# ㄹㅇ 최종중복리스트에서 0번 인덱스만 빼고 다 모음(제거할 기사 인덱스들을 모은 것임)
for i in sim_final_over2:
    final_real += i[1::]

list(set(final_real))

final_real.reverse()

df = df.drop([df.index[i] for i in final_real])

df.to_excel("18_5_final.xlsx",index=False)
