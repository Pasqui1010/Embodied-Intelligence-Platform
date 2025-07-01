# **Quick Start Guide - Embodied Intelligence Platform**

## **🚀 Get Running in 30 Minutes**

This guide gets you from zero to running your first LLM-guided robot simulation.

## **Prerequisites**
- Linux (Ubuntu 22.04+) or Windows with WSL2
- Docker and Docker Compose
- 8GB+ RAM, GPU recommended but not required
- Git

## **Step 1: Clone and Setup (5 minutes)**

```bash
# Clone the repository
git clone --recursive https://github.com/your-org/embodied-intelligence-platform.git
cd embodied-intelligence-platform

# Run automated setup
./scripts/setup_dev_env.sh

# This will:
# - Check dependencies
# - Build Docker images  
# - Create directory structure
# - Setup development environment
```

## **Step 2: Configure API Keys (2 minutes)**

```bash
# Edit .env file
nano .env

# Add your API keys (optional for basic demos):
# OPENAI_API_KEY=your_key_here
# HF_TOKEN=your_huggingface_token
```

## **Step 3: Run Your First Demo (5 minutes)**

### **Option A: Basic Semantic SLAM**
```bash
# Start semantic SLAM demo
docker-compose up demo-slam

# Open browser to http://localhost:3000 to see visualization
# Watch robot build semantic map in real-time
```

### **Option B: LLM-Guided Navigation** (requires API keys)
```bash
# Start full LLM integration demo
docker-compose up demo-full-stack

# Try commands like:
# "Go to the kitchen and find a cup"
# "Navigate to the living room safely"
```

## **Step 4: Interactive Development (10 minutes)**

```bash
# Start development environment
docker-compose up dev-env

# In another terminal, connect to container
docker exec -it embodied-intelligence-platform_dev-env_1 bash

# Build the workspace
cd /workspace
colcon build

# Source the workspace
source install/setup.bash

# Run individual components
ros2 launch eip_slam basic_slam_demo.launch.py
```

## **Step 5: Explore and Contribute (8 minutes)**

### **Browse Examples**
```bash
# Look at progressive complexity examples
ls examples/
# 01_basic_slam/        - Start here
# 02_simple_commands/    - Basic LLM integration  
# 03_proactive_assistance/ - Social behaviors
# 04_continuous_learning/  - Adaptive learning
```

### **Run Tests**
```bash
# Safety tests (critical - must pass)
python -m pytest benchmarks/safety_benchmarks/ -v

# Component tests
colcon test --packages-select eip_slam

# Integration tests
python -m pytest benchmarks/integration_tests/
```

### **Check Documentation**
```bash
# Start documentation server
docker-compose up docs

# Visit http://localhost:8080
```

## **Common Issues & Solutions**

### **Docker Permission Issues**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

### **Display Issues (GUI)**
```bash
# Linux: Allow Docker to access display
xhost +local:docker

# Windows: Use VcXsrv or similar X11 server
```

### **Memory Issues**
```bash
# Increase Docker memory limit to 8GB+
# Docker Desktop: Settings > Resources > Memory
```

### **Build Failures**
```bash
# Clean build
docker-compose down
docker system prune -f
./scripts/setup_dev_env.sh
```

## **Next Steps**

### **For Researchers**
1. **Read**: `docs/architecture/overview.md`
2. **Implement**: Choose a component from the roadmap
3. **Test**: Add benchmarks for your component
4. **Contribute**: Submit PR with tests and documentation

### **For Engineers**
1. **Deploy**: Try real robot integration in `hardware/`
2. **Optimize**: Profile and improve performance
3. **Scale**: Test multi-robot scenarios
4. **Production**: Use deployment pipeline

### **For Students**
1. **Learn**: Complete all example tutorials
2. **Experiment**: Modify safety constraints
3. **Research**: Read papers in `docs/research_papers/`
4. **Build**: Create your own robot application

## **Getting Help**

- **Immediate Help**: GitHub Discussions
- **Bug Reports**: GitHub Issues  
- **Chat**: Discord server (link in README)
- **Video Tutorials**: YouTube playlist
- **Office Hours**: Weekly community calls

## **Contributing Your First Fix**

1. **Find an Issue**: Look for "good first issue" labels
2. **Fork and Clone**: Standard GitHub workflow
3. **Make Changes**: Follow coding standards
4. **Test**: Ensure all tests pass
5. **Document**: Update relevant docs
6. **Submit PR**: Use PR template

## **Component Overview**

```
🏗️ Architecture Layers
├── 🔧 Core (SLAM, Navigation, Manipulation)
├── 🧠 Intelligence (LLM, VLM, Planning) 
├── 🤝 Social (HRI, Emotion, Proactive)
├── 📚 Learning (Adaptation, Validation)
└── 🛡️ Safety (Monitoring, Arbitration)
```

Each layer can be developed independently while maintaining integration compatibility.

## **Development Workflow**

```bash
# 1. Start with safety-verified foundation
docker-compose up demo-slam

# 2. Add intelligence layer
# Modify intelligence/eip_llm_interface/

# 3. Test safety integration
python -m pytest benchmarks/safety_benchmarks/

# 4. Validate in simulation
docker-compose up demo-full-stack

# 5. Deploy to hardware (when ready)
# Follow hardware/deployment/ guides
```

## **Performance Expectations**

| Component | Target Performance | Hardware |
|-----------|-------------------|----------|
| SLAM | 30 FPS, <2cm accuracy | CPU |
| LLM Planning | <500ms response | GPU |
| Safety Monitor | <100ms verification | CPU |
| Full System | Real-time operation | GPU + CPU |

## **Success Checklist**

- [ ] All demos run successfully
- [ ] Safety tests pass (critical)
- [ ] Documentation builds without errors
- [ ] Can modify and rebuild components
- [ ] Understand architecture layers
- [ ] Connected to community channels

**🎉 Congratulations! You're now ready to contribute to the future of intelligent robotics.**

---

**Need help?** Don't hesitate to ask in GitHub Discussions or Discord. The community is here to help you succeed! 