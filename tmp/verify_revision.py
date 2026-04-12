import subprocess
import json

def verify_active_revision():
    try:
        res = subprocess.run(
            ['gcloud', 'run', 'services', 'describe', 'ai-stock-engine', 
             '--format=json', '--region=us-central1', '--project=project-b7ab4ad8-6cf4-491a-b4e'],
            capture_output=True, text=True, shell=True
        )
        if res.returncode != 0:
            return f"Error describing service: {res.stderr}"
        
        data = json.loads(res.stdout)
        latest_ready = data.get('status', {}).get('latestReadyRevisionName')
        traffic = data.get('status', {}).get('traffic', [])
        serving_revision = next((t['revisionName'] for t in traffic if t.get('percent', 0) > 0), None)
        
        # Get env for the latest revision template
        envs = data['spec']['template']['spec']['containers'][0]['env']
        db_url = next((e['value'] for e in envs if e['name'] == 'DATABASE_URL'), None)
        
        return {
            "latest_ready": latest_ready,
            "serving_revision": serving_revision,
            "db_url": db_url,
            "creation_time": data['metadata'].get('creationTimestamp')
        }
    except Exception as e:
        return f"Exception: {e}"

if __name__ == "__main__":
    import pprint
    pprint.pprint(verify_active_revision())
