import pickle as pkl
import json
import random
import sys
sys.path.append(".\\cid\\")

"""
with open("./database/hit_list.pkl","wb") as hitl:
       hit_list = set(random.sample(range(1,15),5))
       print(hit_list)
       pkl.dump(list(hit_list),hitl)
with open("./database/ordinary_list.pkl","wb") as ordl:
       locations = ["Jhansi","Moradabad"]
       temp = list({i for i in range(1,16)}-hit_list)
       ord_list={}
       for loc,pid in zip(locations,range(0,len(temp),5)):
           ord_list[loc]=temp[pid:pid+5]
       print(ord_list)
       pkl.dump(ord_list,ordl)
"""

ids = [i for i in range(1,16)]
names = ["Aaron Eckhart","Aaron Guiel","Aaron Patterson",
         "Aaron Peirsol","Aaron Pena","Aaron Sorkin",
         "Aaron Tippin","Abba Eban","Abbas Kiarostami",
         "Bruna Colosio","Kelly Leigh","Mira Sorvino",
         "Nikki Reed","Princess Victoria","Tamara Brooks"]
addresses=["Jhansi"]*5+["Gwalior"]*5+["Moradabad"]*5
ages=random.sample(range(30,50),15)
gender=["Male"]*8+["Female"]*7
profiles={}
with open("./database/profiles.json","w") as prfls:
     for data in zip(ids,names,addresses,ages,gender):
         profiles[data[0]]={
                "name":data[1],
                "address":data[2],
                "age":data[3],
                "gender":data[4],
                "crimes":0
         }
     json.dump(profiles,prfls)
