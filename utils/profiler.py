import json
import pickle as pkl
import sys
sys.path.append(".\\utils\\")

def new_criminal(location,nw_id):
    with open("crime_profiles.json","w") as cp:
        data1 = json.load(cp)
        with open("public_profiles.json","w") as pp:
           data2 = json.load(pp)
           data1[nw_id]=data2[location][nw_id]
           del data2[nw_id]
           json.dump(data2,pp)
        json.dump(data1,pp)

def fetch_details(cr_id):
   with open("crime_profiles.json","r") as cp:
        data = json.load(cp)
        try:
            return data[cr_id]
        except:
            new_criminal(cr_id)
            data = json.load(cp)
            return data[cr_id]

def managehitlist():
    with open("crime_profiles.json","w") as cp:
        data1 = json.load(cp)
        with open ("hit_list.pkl","w") as hl:
            data2 = sorted(data1,lambda x:data1[x]["crime_count"])
            pkl.dump(data2,hl)
        
