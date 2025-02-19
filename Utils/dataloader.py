from glob import glob
import pandas as pd 
import os 

def load_subs_datasets(dataset_path):
    subtitles_paths = glob(dataset_path + '/*.ass')

    scripts = []
    episode_num = []
    seasons = []

    for path in subtitles_paths:
        filename = os.path.basename(path)
        
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [",".join(line.split(',')[9:]) for line in lines]
        
        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)
        
        left_side = filename.split('-')[0].strip()  # "Naruto Season 1"
        # Sezon numarasını çek
        # Örneğin left_side.split() -> ["Naruto", "Season", "1"]
        season_num = int(left_side.split()[-1])
        episode = int(filename.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)
        seasons.append(season_num)  # direkt rakam olarak ekliyoruz

    df = pd.DataFrame({
        "episode": episode_num,
        "script": scripts
    })

    df.sort_values(["episode"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df