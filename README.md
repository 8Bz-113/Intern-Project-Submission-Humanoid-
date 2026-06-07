# Humanoid AI Intern Project Submission

GitHub repository for **Abhinav Mayura's application to Humanoid AI**.

## Requirements

- Ubuntu
- NVIDIA Isaac Sim installed at:

```text
~/isaacsim
```

To Clone the Git Repository:
```text
cd ~
git clone https://github.com/8Bz-113/Intern-Project-Submission-Humanoid-.git
```
This will create a folder called 'Intern-Project-Submission-Humanoid-' in your home directory

Then to run the script:
```text
cd ~/Intern-Project-Submission-Humanoid- 
~/isaacsim/python.sh scripts/end_effector_tracking.py
```
Running the script will create a folder called rl_orbit_logs, which contains csv. files containg data for each episode
It will also create a file called episode_summary.csv which contains high level episode-by-episode data for the simulation
