import subprocess
import json

def get_db_url():
    try:
        # Run gcloud command
        res = subprocess.run(
            ['gcloud', 'run', 'services', 'describe', 'ai-stock-engine', 
             '--format=json', '--region=us-central1', '--project=project-b7ab4ad8-6cf4-491a-b4e'],
            capture_output=True, text=True, shell=True
        )
        if res.returncode != 0:
            print(f"Error: {res.stderr}")
            return None
        
        data = json.loads(res.stdout)
        # Handle both service list and single service item formats
        if isinstance(data, list):
            data = data[0]
            
        envs = data['spec']['template']['spec']['containers'][0]['env']
        return next((e['value'] for e in envs if e['name'] == 'DATABASE_URL'), None)
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    url = get_db_url()
    if url:
        print(url)
