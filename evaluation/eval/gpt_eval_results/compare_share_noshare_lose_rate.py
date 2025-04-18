import json


# open the json file
def open_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    
    metrics = ["closeness", "coherence", "engagingness", "reflectiveness"]

    for metric in metrics:
        acuum_path = f"accum_0927_4_session_{metric}.json"
        accum_dataset = open_json_file(acuum_path)
        update_path = f"update_0927_4_session_{metric}.json"
        update_dataset = open_json_file(update_path)
        total = 0
        lose = 0
    
        for data_key, data_value in accum_dataset.items():
            accum_score = data_value[metric]
            update_score = update_dataset[data_key][metric]
            if accum_score > update_score:
                lose += 1
            total += 1
        
        print(f"metric: {metric}")
        lose_rate = lose/total
        print(f"lose rate: {(lose_rate):0.4f}")
        print(f"win and tie rate: {(1-lose_rate):0.4f}")
            
        

if __name__ == '__main__':
    main()