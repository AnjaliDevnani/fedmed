import json
import os
from config import cfg

def audit():
    history_file = os.path.join(cfg.results_dir, "fl_history.json")
    if not os.path.exists(history_file):
        print(f"Privacy audit failed: {history_file} not found.")
        return
        
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
            
        epsilons = data.get('privacy_epsilon', [])
        if epsilons:
            final_epsilon = epsilons[-1]
            print(f"\n{'='*40}")
            print("PRIVACY AUDIT REPORT")
            print(f"{'='*40}")
            print(f"Target Epsilon: {cfg.dp_target_epsilon}")
            print(f"Actual Epsilon Spent: {final_epsilon:.4f}")
            print(f"Target Delta: {cfg.dp_target_delta}")
            
            if final_epsilon <= cfg.dp_target_epsilon:
                print("Status: PASSED (Within Budget)")
            else:
                print("Status: FAILED (Exceeded Budget)")
            print(f"{'='*40}\n")
        else:
            print("No privacy tracking data found in history.")
            
    except Exception as e:
        print(f"Error reading privacy history: {e}")

if __name__ == "__main__":
    audit()
