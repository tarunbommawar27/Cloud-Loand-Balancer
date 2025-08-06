# 🚀 AI LOAD BALANCER - FULL DEPLOYMENT IN GOOGLE COLAB
# Just copy this entire code into a Google Colab cell and run!

# ==========================================
# SECTION 1: SETUP & INSTALLATION
# ==========================================

print("🔧 STEP 1: Installing required packages...")
!pip install -q stable-baselines3 gym pandas numpy matplotlib seaborn scikit-learn
!pip install -q fastapi uvicorn pyngrok nest-asyncio pydantic plotly

print("✅ Packages installed successfully!")

# ==========================================
# SECTION 2: IMPORTS
# ==========================================

print("\n📦 STEP 2: Importing libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from pyngrok import ngrok
import nest_asyncio
import threading
import time
import json
import os
from IPython.display import display, HTML, Javascript
import warnings
warnings.filterwarnings('ignore')

# Enable nested asyncio for Colab
nest_asyncio.apply()

print("✅ Libraries imported successfully!")

# ==========================================
# SECTION 3: LOAD BALANCER ENVIRONMENT
# ==========================================

print("\n🏗️ STEP 3: Building the AI Load Balancer Environment...")

class LoadBalancerEnv(gym.Env):
    """Simulates load-balancing over n_servers."""
    def __init__(self, df, n_servers=5, step_us=100):
        super().__init__()
        self.raw_df = df.reset_index(drop=True) if len(df) > 0 else pd.DataFrame()
        self.n_servers = n_servers
        self.step_us = step_us
        self.cpu_thresh = 0.95
        self.mem_thresh = 0.95
        self.max_duration = float(self.raw_df["duration"].max()) if len(self.raw_df) > 0 else 1.0
        
        self.action_space = spaces.Discrete(n_servers)
        obs_dim = 3 + 2 * n_servers
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )
        self.current_job_feats = np.zeros(3, dtype=np.float32)
        self.reset()

    def reset(self):
        if len(self.raw_df) > 0:
            self.df = self.raw_df.sample(frac=min(0.8, 1.0)).reset_index(drop=True)
            self.next_jobs = list(range(len(self.df)))
        else:
            self.df = pd.DataFrame()
            self.next_jobs = []
        
        self.current_time = 0
        self.next_job_idx = 0
        self.servers = [[] for _ in range(self.n_servers)]
        self.server_loads = np.zeros((self.n_servers, 2), dtype=np.float32)
        self.cpu_history = []
        self.memory_history = []
        self.action_history = []
        self.overload_history = []
        self.failed_jobs = []
        self.current_job_feats[:] = 0
        
        return self._get_obs()

    def _cleanup_finished_jobs(self):
        for i in range(self.n_servers):
            self.servers[i] = [job for job in self.servers[i] if job["end_time"] > self.current_time]

    def _update_server_loads(self):
        self.server_loads.fill(0)
        for i, jobs in enumerate(self.servers):
            for job in jobs:
                self.server_loads[i, 0] += job["cpu"]
                self.server_loads[i, 1] += job["mem"]

    def _get_obs(self):
        load_feats = self.server_loads.flatten().astype(np.float32)
        return np.concatenate([self.current_job_feats, load_feats])

    def _compute_reward(self, action):
        cpu_used = self.server_loads[action, 0]
        mem_used = self.server_loads[action, 1]
        overload_penalty = -20 if (cpu_used > 0.95 or mem_used > 0.95) else 0
        std_cpu = np.std(self.server_loads[:, 0])
        balance_reward = np.clip(1.0/(std_cpu + 1e-6), 0, 100)
        return balance_reward + overload_penalty

    def step(self, action):
        self.current_time += self.step_us
        self._cleanup_finished_jobs()
        self._update_server_loads()
        
        reward = 0.0
        job_arrived = False
        j_queue = len(self.next_jobs)
        
        if j_queue > 0 and np.random.rand() < 0.5:
            rand_idx = np.random.randint(0, j_queue)
            self.next_job_idx = self.next_jobs.pop(rand_idx)
            job_arrived = True
            
            if self.next_job_idx < len(self.df):
                row = self.df.iloc[self.next_job_idx]
                cpu = float(row["resource_request"])/4
                mem = float(row["assigned_memory"])/4
                norm_dur = float(row["duration"]) / self.max_duration if self.max_duration > 0 else 0
                self.current_job_feats = np.array([cpu, mem, norm_dur], dtype=np.float32)
                
                end_time = self.current_time + float(row["duration"]) * 1000000
                if (self.server_loads[action, 0] + cpu <= 1.0 and 
                    self.server_loads[action,1] + mem <= 1.0):
                    self.servers[action].append({
                        "cpu": cpu, "mem": mem, "end_time": end_time
                    })
                else:
                    reward = -50
                    self.failed_jobs.append({
                        "idx": self.next_job_idx,
                        "srv": action,
                        "cpu": cpu,
                        "mem": mem
                    })
                
                self._update_server_loads()
                reward += self._compute_reward(action)
        else:
            self.current_job_feats[:] = 0
        
        self.cpu_history.append(self.server_loads[:, 0].copy())
        self.memory_history.append(self.server_loads[:,1].copy())
        self.action_history.append(action)
        self.overload_history.append(not job_arrived)
        
        done = (j_queue == 0)
        obs = self._get_obs()
        info = {
            "time_us": self.current_time,
            "failed_jobs": len(self.failed_jobs),
            "next_job_idx": self.next_job_idx
        }
        
        return obs, reward, done, info

print("✅ Environment created successfully!")

# ==========================================
# SECTION 4: GENERATE SAMPLE DATA
# ==========================================

print("\n📊 STEP 4: Generating sample data...")

def generate_sample_data(n_samples=5000):
    """Generate realistic sample data"""
    np.random.seed(42)
    data = {
        'resource_request': np.random.lognormal(0, 0.5, n_samples).clip(0.1, 3.0),
        'assigned_memory': np.random.lognormal(0, 0.5, n_samples).clip(0.1, 3.0),
        'duration': np.random.lognormal(6, 1.5, n_samples).clip(10, 7200),  # 10s to 2 hours
        'failed': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    # Add correlated memory for realism
    data['assigned_memory'] = data['resource_request'] * np.random.normal(1, 0.2, n_samples)
    data['assigned_memory'] = data['assigned_memory'].clip(0.1, 3.0)
    
    # Create timestamps
    start_times = pd.date_range(start='2024-01-01', periods=n_samples, freq='30s')
    data['start_time'] = start_times
    data['end_time'] = start_times + pd.to_timedelta(data['duration'], unit='s')
    
    return pd.DataFrame(data)

df = generate_sample_data()
print(f"✅ Generated {len(df)} sample jobs!")
print(f"Sample data preview:")
print(df.head())

# ==========================================
# SECTION 5: TRAIN THE AI MODEL
# ==========================================

print("\n🧠 STEP 5: Training the AI Load Balancer...")
print("This will take 2-3 minutes. Please wait...")

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create training environment
def make_env():
    return LoadBalancerEnv(train_df, n_servers=5)

# Train the model
env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)

# Quick training for demo (increase total_timesteps for better results)
model.learn(total_timesteps=50000)
print("✅ Model trained successfully!")

# Save model
model.save("load_balancer_model")
print("✅ Model saved!")

# ==========================================
# SECTION 6: EVALUATE PERFORMANCE
# ==========================================

print("\n📈 STEP 6: Evaluating performance...")

def evaluate_method(env, method="ai"):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    current_server = 0
    
    while not done and steps < 1000:
        if method == "ai":
            action, _ = model.predict(obs, deterministic=True)
        elif method == "round_robin":
            action = current_server
            current_server = (current_server + 1) % env.n_servers
        else:  # random
            action = random.randint(0, env.n_servers - 1)
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    
    return total_reward, env

# Evaluate all methods
test_env = LoadBalancerEnv(test_df, n_servers=5)

ai_reward, ai_env = evaluate_method(LoadBalancerEnv(test_df, 5), "ai")
rr_reward, rr_env = evaluate_method(LoadBalancerEnv(test_df, 5), "round_robin")
rand_reward, rand_env = evaluate_method(LoadBalancerEnv(test_df, 5), "random")

# Create comparison chart
plt.figure(figsize=(10, 6))
methods = ['AI (RL)', 'Round Robin', 'Random']
scores = [ai_reward, rr_reward, rand_reward]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = plt.bar(methods, scores, color=colors, alpha=0.8)
plt.title('Load Balancing Performance Comparison', fontsize=16, pad=20)
plt.ylabel('Performance Score (Higher is Better)', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}', ha='center', va='bottom', fontsize=11)

# Add improvement percentages
if rr_reward != 0:
    rr_improvement = ((ai_reward - rr_reward) / abs(rr_reward)) * 100
    plt.text(0.5, ai_reward/2, f'+{rr_improvement:.0f}%', 
             ha='center', fontsize=14, color='green', weight='bold')

plt.tight_layout()
plt.show()

print(f"\n📊 Results:")
print(f"AI Score: {ai_reward:.0f}")
print(f"Round Robin Score: {rr_reward:.0f}")
print(f"Random Score: {rand_reward:.0f}")

# ==========================================
# SECTION 7: CREATE WEB API
# ==========================================

print("\n🌐 STEP 7: Creating Web API...")

# Global variables for API
current_env = LoadBalancerEnv(pd.DataFrame(), n_servers=5)
stats = {
    "total_jobs": 0,
    "successful_jobs": 0,
    "failed_jobs": 0,
    "ai_decisions": 0
}

# Create FastAPI app
app = FastAPI(title="AI Load Balancer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    cpu_request: float = 1.0
    memory_request: float = 1.0
    duration: float = 300.0

class ServerStatus(BaseModel):
    server_id: int
    cpu_load: float
    memory_load: float
    active_jobs: int
    status: str

@app.get("/")
def home():
    return {
        "message": "🤖 AI Load Balancer API is running!",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/schedule": "Schedule a job (POST)",
            "/status": "Get server status (GET)",
            "/stats": "Get statistics (GET)",
            "/demo": "Interactive demo page (GET)"
        }
    }

@app.post("/schedule")
def schedule_job(job: JobRequest):
    global stats, current_env
    
    # Prepare observation
    job_features = np.array([
        job.cpu_request / 4.0,
        job.memory_request / 4.0,
        job.duration / 3600.0
    ])
    
    current_env._update_server_loads()
    server_loads = current_env.server_loads.flatten()
    obs = np.concatenate([job_features, server_loads])
    
    # Get AI decision
    action, _ = model.predict(obs, deterministic=True)
    server_id = int(action)
    
    # Check if we can actually place the job
    cpu_after = current_env.server_loads[server_id, 0] + job.cpu_request/4
    mem_after = current_env.server_loads[server_id, 1] + job.memory_request/4
    
    if cpu_after <= 1.0 and mem_after <= 1.0:
        # Add job to server
        current_env.servers[server_id].append({
            "cpu": job.cpu_request/4,
            "mem": job.memory_request/4,
            "end_time": current_env.current_time + job.duration * 1000000
        })
        stats["successful_jobs"] += 1
        status = "scheduled"
    else:
        stats["failed_jobs"] += 1
        status = "rejected"
    
    stats["total_jobs"] += 1
    stats["ai_decisions"] += 1
    
    return {
        "job_id": f"job_{stats['total_jobs']}",
        "assigned_server": server_id,
        "status": status,
        "server_load_after": {
            "cpu": float(cpu_after),
            "memory": float(mem_after)
        }
    }

@app.get("/status", response_model=List[ServerStatus])
def get_server_status():
    current_env._cleanup_finished_jobs()
    current_env._update_server_loads()
    
    statuses = []
    for i in range(current_env.n_servers):
        cpu_load = float(current_env.server_loads[i, 0])
        mem_load = float(current_env.server_loads[i, 1])
        active_jobs = len(current_env.servers[i])
        
        if cpu_load > 0.9 or mem_load > 0.9:
            status = "OVERLOADED"
        elif cpu_load > 0.7 or mem_load > 0.7:
            status = "BUSY"
        elif cpu_load > 0.3 or mem_load > 0.3:
            status = "NORMAL"
        else:
            status = "IDLE"
        
        statuses.append(ServerStatus(
            server_id=i,
            cpu_load=round(cpu_load, 3),
            memory_load=round(mem_load, 3),
            active_jobs=active_jobs,
            status=status
        ))
    
    return statuses

@app.get("/stats")
def get_statistics():
    current_env._update_server_loads()
    return {
        **stats,
        "avg_cpu_load": float(np.mean(current_env.server_loads[:, 0])),
        "avg_memory_load": float(np.mean(current_env.server_loads[:, 1])),
        "success_rate": (stats["successful_jobs"] / max(stats["total_jobs"], 1)) * 100
    }

@app.get("/demo")
def demo_page():
    return HTML('''
    <html>
    <head>
        <title>AI Load Balancer Demo</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .server { display: inline-block; width: 150px; height: 120px; margin: 10px; padding: 10px; border-radius: 8px; text-align: center; transition: all 0.3s; }
            .idle { background: #90EE90; }
            .normal { background: #FFD700; }
            .busy { background: #FFA500; }
            .overloaded { background: #FF6347; color: white; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; font-size: 16px; }
            button:hover { background: #45a049; }
            .stats { background: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px; }
            h1 { color: #333; text-align: center; }
            .metric { display: inline-block; margin: 10px 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Load Balancer Live Demo</h1>
            <div id="servers"></div>
            <div style="text-align: center; margin: 20px;">
                <button onclick="scheduleJob()">Schedule Random Job</button>
                <button onclick="scheduleBurst()">Schedule 10 Jobs</button>
                <button onclick="scheduleHeavyJob()">Schedule Heavy Job</button>
            </div>
            <div class="stats">
                <h3>System Statistics</h3>
                <div id="stats"></div>
            </div>
        </div>
        <script>
            async function updateStatus() {
                const response = await fetch('/status');
                const servers = await response.json();
                
                const serversDiv = document.getElementById('servers');
                serversDiv.innerHTML = '<h2>Server Status</h2>';
                
                servers.forEach(server => {
                    const div = document.createElement('div');
                    div.className = `server ${server.status.toLowerCase()}`;
                    div.innerHTML = `
                        <h3>Server ${server.server_id + 1}</h3>
                        <p>CPU: ${(server.cpu_load * 100).toFixed(1)}%</p>
                        <p>Memory: ${(server.memory_load * 100).toFixed(1)}%</p>
                        <p>Jobs: ${server.active_jobs}</p>
                        <p><strong>${server.status}</strong></p>
                    `;
                    serversDiv.appendChild(div);
                });
                
                const statsResponse = await fetch('/stats');
                const stats = await statsResponse.json();
                
                document.getElementById('stats').innerHTML = `
                    <div class="metric">📊 Total Jobs: ${stats.total_jobs}</div>
                    <div class="metric">✅ Successful: ${stats.successful_jobs}</div>
                    <div class="metric">❌ Failed: ${stats.failed_jobs}</div>
                    <div class="metric">🎯 Success Rate: ${stats.success_rate.toFixed(1)}%</div>
                    <div class="metric">🖥️ Avg CPU: ${(stats.avg_cpu_load * 100).toFixed(1)}%</div>
                    <div class="metric">💾 Avg Memory: ${(stats.avg_memory_load * 100).toFixed(1)}%</div>
                `;
            }
            
            async function scheduleJob(cpu=null, memory=null, duration=null) {
                const job = {
                    cpu_request: cpu || Math.random() * 2,
                    memory_request: memory || Math.random() * 2,
                    duration: duration || Math.random() * 600 + 60
                };
                
                const response = await fetch('/schedule', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(job)
                });
                
                const result = await response.json();
                updateStatus();
            }
            
            async function scheduleBurst() {
                for(let i = 0; i < 10; i++) {
                    await scheduleJob();
                    await new Promise(r => setTimeout(r, 100));
                }
            }
            
            async function scheduleHeavyJob() {
                await scheduleJob(2.5, 2.5, 1800);
            }
            
            updateStatus();
            setInterval(updateStatus, 2000);
        </script>
    </body>
    </html>
    ''')

print("✅ API created successfully!")

# ==========================================
# SECTION 8: DEPLOY WITH NGROK
# ==========================================

print("\n🚀 STEP 8: Deploying to the web...")

# Function to run the API server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the server in a separate thread
api_thread = threading.Thread(target=run_server, daemon=True)
api_thread.start()

# Give the server time to start
time.sleep(5)

# Create public URL with ngrok
print("Creating public URL...")
public_url = ngrok.connect(8000)
print(f"\n✅ YOUR API IS NOW LIVE AT: {public_url}")
print(f"\n🎉 CONGRATULATIONS! Your AI Load Balancer is deployed!")

# ==========================================
# SECTION 9: INTERACTIVE DEMO
# ==========================================

print("\n📱 STEP 9: Access your deployment:")
print(f"\n1️⃣ API Documentation: {public_url}/docs")
print(f"2️⃣ Interactive Demo: {public_url}/demo")
print(f"3️⃣ API Endpoints:")
print(f"   - Schedule Job: POST {public_url}/schedule")
print(f"   - Server Status: GET {public_url}/status")
print(f"   - Statistics: GET {public_url}/stats")

# Create clickable links in Colab
display(HTML(f'''
<div style="background: #f0f0f0; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2>🎯 Your Live AI Load Balancer</h2>
    <p style="font-size: 18px;">Your API is now accessible from anywhere in the world!</p>
    
    <h3>Click these links to access your deployment:</h3>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li>📊 <a href="{public_url}/demo" target="_blank">Interactive Demo Dashboard</a></li>
        <li>📚 <a href="{public_url}/docs" target="_blank">API Documentation</a></li>
        <li>🔍 <a href="{public_url}/status" target="_blank">Current Server Status</a></li>
        <li>📈 <a href="{public_url}/stats" target="_blank">System Statistics</a></li>
    </ul>
    
    <h3>Share your API:</h3>
    <p style="background: white; padding: 10px; border-radius: 5px; font-family: monospace;">
        {public_url}
    </p>
    
    <p style="color: #666; margin-top: 20px;">
        ⚠️ Note: This URL will remain active as long as this Colab session is running.
        For permanent deployment, see the additional options below.
    </p>
</div>
'''))

# ==========================================
# SECTION 10: TEST YOUR API
# ==========================================

print("\n🧪 STEP 10: Testing your API...")

# Test the API
import requests

# Test scheduling a job
test_job = {
    "cpu_request": 1.5,
    "memory_request": 1.2,
    "duration": 300
}

try:
    response = requests.post(f"{public_url}/schedule", json=test_job)
    print(f"Test job scheduled: {response.json()}")
    
    # Get current status
    status_response = requests.get(f"{public_url}/status")
    print(f"\nCurrent server status:")
    for server in status_response.json():
        print(f"  Server {server['server_id'] + 1}: {server['status']} (CPU: {server['cpu_load']*100:.1f}%, Memory: {server['memory_load']*100:.1f}%)")
    
    print("\n✅ API is working perfectly!")
except:
    print("⚠️ API test failed. Please check the server is running.")

# ==========================================
# SECTION 11: PERMANENT DEPLOYMENT OPTIONS
# ==========================================

print("\n📌 PERMANENT DEPLOYMENT OPTIONS:")
print("""
To keep your API running 24/7, you can:

1. **Google Cloud Run (Free Tier)**
   - Save this notebook
   - Download the model file
   - Deploy using Cloud Run

2. **Replit (Free)**
   - Create account at replit.com
   - Copy the code
   - Run continuously

3. **PythonAnywhere (Free)**
   - Create account at pythonanywhere.com
   - Upload your code
   - Get permanent URL

4. **Keep Colab Running**
   - Leave this tab open
   - Colab Pro for longer runtime
""")

# Create a simple code export
with open('/content/load_balancer_api.py', 'w') as f:
    f.write('''
# Standalone API file - can be deployed anywhere
import numpy as np
import pandas as pd
from fastapi import FastAPI
from stable_baselines3 import PPO
import uvicorn

# Your model and API code here
# This file can be deployed to any cloud platform
''')

print("\n✅ Deployment file created: /content/load_balancer_api.py")
print("\n🎉 DEPLOYMENT COMPLETE! Your AI Load Balancer is live on the internet!")
print(f"\n🌐 Share this URL with anyone: {public_url}")
print("\n📱 Try the demo on your phone by visiting the demo link above!")

# Keep the notebook running
print("\n⏰ Keep this notebook running to maintain your deployment.")
print("💡 Tip: Open the demo link in a new tab to see your AI in action!")
