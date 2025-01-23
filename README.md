# **GRPOptim** 🚀  
**Group Relative Policy Optimization for Efficient Reinforcement Learning**  

[![PyPI Version](https://img.shields.io/pypi/v/GRPOptim.svg)](https://pypi.org/project/grpoptim/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
[![Python Versions](https://img.shields.io/pypi/pyversions/GRPOptim.svg)](https://pypi.org/project/grpoptim/)  
[![Downloads](https://static.pepy.tech/badge/GRPOptim/month)](https://pepy.tech/project/grpoptim)  
[![Build Status](https://github.com/subaashnair/GRPOptim/actions/workflows/tests.yml/badge.svg)](https://github.com/subaashnair/grpoptim/actions)  

---

### **What is GRPOptim?**  
**GRPOptim** is a lightweight Python library for training reinforcement learning (RL) agents using **Group Relative Policy Optimization (GRPO)**, a critic-free algorithm that reduces training costs by leveraging group-based advantage estimation and KL regularization. Ideal for tasks like language model alignment, robotics, and game AI.  

---

### **Key Features**  
- 🚫 **No Critic Model**: Trains policies directly with 50% fewer parameters.  
- 📊 **Group-Based Advantage**: Stabilizes training using reward standardization.  
- 🔧 **PyTorch/TensorFlow Compatible**: Easy integration with your ML pipeline.  
- 🐍 **Pythonic API**: Minimal boilerplate for RL practitioners.  

---

### **Installation**  
```bash  
pip install GRPOptim  
```  

---

### **Quick Example**  
```python  
import GRPOptim as gro  

# Initialize GRPO trainer  
trainer = gro.GRPO(  
    policy_model=your_policy_network,  
    reference_model=your_reference_network,  
    epsilon=0.2,  
    beta=0.1  
)  

# Train with group sampling  
for epoch in range(epochs):  
    group = trainer.sample_group(inputs)  
    rewards = compute_rewards(group)  
    loss = trainer.step(group, rewards)  
```  

---

### **Why GRPOptim?**  
- 🏎️ **Faster Training**: Skip the critic network and train policies directly.  
- 📉 **Stable Convergence**: Group normalization and clipping prevent reward collapse.  
- 🧩 **Drop-In Replacement**: Compatible with RL workflows (PPO, A2C, etc.).  

---

### **License**  
MIT License. See [LICENSE](LICENSE).  

---

### **Name Justification**  
- **GRPO**: Direct reference to the algorithm.  
- **Optim**: Short for "optimization," emphasizing efficiency.  
- **Unique & Memorable**: Easy to search for on PyPI/GitHub.  

---