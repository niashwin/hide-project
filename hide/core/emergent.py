"""
Emergent memory phenomena: DRM false memory, spacing effect, tip-of-tongue, topology.
"""

import numpy as np
from typing import Dict, List, Tuple

# All 24 DRM word lists (Roediger & McDermott 1995, public domain)
DRM_LISTS = {
    "SLEEP": {"studied": ["bed","rest","awake","tired","dream","wake","snooze","blanket","doze","slumber","snore","nap","peace","yawn","drowsy"], "lure": "sleep"},
    "NEEDLE": {"studied": ["thread","pin","eye","sewing","sharp","point","prick","thimble","haystack","thorn","hurt","injection","syringe","cloth","knitting"], "lure": "needle"},
    "ROUGH": {"studied": ["smooth","bumpy","road","tough","sandpaper","jagged","ready","coarse","uneven","riders","rugged","sand","boards","ground","gravel"], "lure": "rough"},
    "SWEET": {"studied": ["sour","candy","sugar","bitter","good","taste","tooth","nice","honey","soda","chocolate","heart","cake","tart","pie"], "lure": "sweet"},
    "CHAIR": {"studied": ["table","sit","legs","seat","couch","desk","recliner","sofa","wood","cushion","swivel","stool","sitting","rocking","bench"], "lure": "chair"},
    "WINDOW": {"studied": ["door","glass","pane","shade","ledge","sill","house","open","curtain","frame","view","breeze","sash","screen","shutter"], "lure": "window"},
    "SMELL": {"studied": ["nose","breathe","sniff","aroma","hear","see","nostril","whiff","scent","reek","stink","fragrance","perfume","salts","rose"], "lure": "smell"},
    "MOUNTAIN": {"studied": ["hill","valley","climb","summit","top","molehill","peak","plain","glacier","goat","bike","climber","range","steep","ski"], "lure": "mountain"},
    "MUSIC": {"studied": ["note","sound","piano","sing","radio","band","melody","horn","concert","instrument","symphony","jazz","orchestra","art","rhythm"], "lure": "music"},
    "COLD": {"studied": ["hot","snow","warm","winter","ice","wet","frigid","chilly","heat","weather","freeze","air","shiver","arctic","frost"], "lure": "cold"},
    "ANGER": {"studied": ["mad","fear","hate","rage","temper","fury","ire","wrath","happy","fight","hatred","mean","calm","emotion","enrage"], "lure": "anger"},
    "DOCTOR": {"studied": ["nurse","sick","lawyer","medicine","health","hospital","dentist","physician","ill","patient","office","stethoscope","surgeon","clinic","cure"], "lure": "doctor"},
    "RIVER": {"studied": ["water","stream","lake","Mississippi","boat","tide","swim","flow","run","barge","creek","brook","fish","bridge","winding"], "lure": "river"},
    "FRUIT": {"studied": ["apple","vegetable","orange","kiwi","citrus","ripe","pear","banana","berry","cherry","basket","juice","salad","bowl","cocktail"], "lure": "fruit"},
    "BLACK": {"studied": ["white","dark","cat","charcoal","night","funeral","color","grief","blue","death","ink","bottom","coal","brown","gray"], "lure": "black"},
    "KING": {"studied": ["queen","England","crown","prince","George","dictator","palace","throne","chess","rule","subjects","monarch","royal","leader","reign"], "lure": "king"},
    "BREAD": {"studied": ["butter","food","eat","sandwich","rye","jam","milk","flour","jelly","dough","crust","slice","wine","loaf","toast"], "lure": "bread"},
    "SPIDER": {"studied": ["web","insect","bug","fright","fly","arachnid","crawl","tarantula","poison","bite","creepy","animal","ugly","feelers","small"], "lure": "spider"},
    "SLOW": {"studied": ["fast","lethargic","stop","listless","snail","cautious","delay","traffic","turtle","hesitant","speed","quick","sluggish","wait","molasses"], "lure": "slow"},
    "MAN": {"studied": ["woman","husband","uncle","lady","mouse","male","father","strong","friend","beard","person","handsome","muscle","suit","old"], "lure": "man"},
    "SOFT": {"studied": ["hard","light","pillow","plush","loud","cotton","fur","touch","fluffy","feather","tender","skin","silk","smooth","kitten"], "lure": "soft"},
    "THIEF": {"studied": ["steal","robber","crook","burglar","money","cop","bad","rob","jail","gun","villain","crime","bank","bandit","criminal"], "lure": "thief"},
    "HIGH": {"studied": ["low","clouds","up","tall","tower","jump","above","building","noon","over","airplane","dive","elevate","cliff","sky"], "lure": "high"},
    "LION": {"studied": ["tiger","circus","jungle","tamer","den","cub","Africa","mane","cage","feline","roar","fierce","wildcat","pride","cougar"], "lure": "lion"},
}


def drm_experiment(encode_fn, threshold: float = 0.82) -> Dict:
    """Run the DRM false memory experiment.

    Args:
        encode_fn: Function that takes a list of strings and returns embeddings (n, dim).
        threshold: Cosine similarity threshold for "recognition".

    Returns:
        Dict with hit_rate, critical_fa_rate, unrelated_fa_rate, per_list results.
    """
    results = {}
    for list_name, data in DRM_LISTS.items():
        studied = data["studied"]
        lure = data["lure"]
        # Unrelated words from a different list
        other_lists = [k for k in DRM_LISTS if k != list_name]
        unrelated = DRM_LISTS[other_lists[0]]["studied"][:5]

        all_words = studied + [lure] + unrelated
        embeddings = encode_fn(all_words)
        studied_embs = embeddings[:len(studied)]
        lure_emb = embeddings[len(studied)]
        unrelated_embs = embeddings[len(studied) + 1:]

        # Cosine similarities
        centroid = studied_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        studied_sims = [float(np.dot(e / np.linalg.norm(e), centroid)) for e in studied_embs]
        lure_sim = float(np.dot(lure_emb / np.linalg.norm(lure_emb), centroid))
        unrelated_sims = [float(np.dot(e / np.linalg.norm(e), centroid)) for e in unrelated_embs]

        results[list_name] = {
            "studied_mean_sim": np.mean(studied_sims),
            "lure_sim": lure_sim,
            "unrelated_mean_sim": np.mean(unrelated_sims),
            "lure_above_threshold": lure_sim > threshold,
        }

    return results
