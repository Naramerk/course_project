import requests
import json
import os
from datetime import datetime, timezone
import argparse

def get_reddit_token(client_id, client_secret, refresh=False):
    """
    Get a Reddit API token using client credentials
    """
    token_file = os.path.join(os.path.dirname(__file__), 'reddit_token.json')
    
    # Check if token exists and is not expired
    if os.path.exists(token_file) and not refresh:
        with open(token_file) as f:
            token_info = json.load(f)
            retrieved_at = datetime.fromisoformat(token_info['retrieved_at'])
            expires_in = token_info['expires_in']
            if datetime.now(timezone.utc) < retrieved_at + timedelta(seconds=expires_in):
                print("Using existing valid token")
                return token_info
    
    # Get new token
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {
        'grant_type': 'client_credentials',
    }
    headers = {
        'User-Agent': 'Political Sentiment Analysis/1.0'
    }

    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data=data,
            headers=headers
        )
        response.raise_for_status()
        
        token_info = response.json()
        token_info['retrieved_at'] = datetime.now(timezone.utc).isoformat()
        
        # Save token
        with open(token_file, 'w') as f:
            json.dump(token_info, f, indent=4)
            
        print("Successfully obtained new token")
        return token_info
        
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Reddit API token")
    parser.add_argument("--client-id", default="eYMJ7Td4dqK3ROGLEWqgZQ", help="Reddit API client ID")
    parser.add_argument("--client-secret", default="BOs69e4HSw_O3Q6Qa1YXAW_S8NwAIA", help="Reddit API client secret")
    parser.add_argument("--refresh", action="store_true", help="Force token refresh")
    args = parser.parse_args()
    
    token_info = get_reddit_token(args.client_id, args.client_secret, args.refresh)
    if token_info:
        print("Token info:", json.dumps(token_info, indent=4)) 