import pandas as pd
import numpy as np

df = pd.read_excel("DATA\ECU_IoHT.xlsx")
data = df.to_dict('records')

MAX_LEN = 1024
def format_packet(pkt_bytes):
    pkt = list(pkt_bytes[:MAX_LEN])
    return pkt + [0]*(MAX_LEN - len(pkt)) if len(pkt) < MAX_LEN else pkt

def normalize(pkt):
    return (np.array(pkt) - 128) / 128.0