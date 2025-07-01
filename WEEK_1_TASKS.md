# **Week 1 Implementation Tasks - Embodied Intelligence Platform**

## **🎯 Goal: Get `docker-compose up demo-slam` Working End-to-End**

### **Day 1: Complete Missing Build Files**

```bash
# Tasks:
1. Create CMakeLists.txt for eip_interfaces
2. Create CMakeLists.txt for eip_slam  
3. Create setup.py for Python packages
4. Fix package.xml name tags (currently has <n> instead of <name>)
5. Test message compilation
```

**Commands to run:**
```bash
# Build and test interfaces
cd Service/
colcon build --packages-select eip_interfaces
source install/setup.bash

# Verify messages are generated
ros2 interface list | grep eip_interfaces
```

### **Day 2: Safety Monitor Testing**

```bash
# Tasks:
1. Complete human detection placeholder
2. Add workspace boundary checking
3. Create safety benchmark test suite
4. Test emergency stop functionality
```

**Test command:**
```bash
# Run safety tests
python -m pytest benchmarks/safety_benchmarks/ -v
```

### **Day 3: SLAM Integration**

```bash
# Tasks:
1. Create CMakeLists.txt for eip_slam
2. Test basic SLAM node compilation
3. Create RViz configuration file
4. Test point cloud processing
```

**Test command:**
```bash
# Build SLAM package
colcon build --packages-select eip_slam
source install/setup.bash

# Test SLAM node
ros2 run eip_slam basic_slam_node.py
```

### **Day 4: Demo Integration**

```bash
# Tasks:
1. Test complete launch file
2. Integrate safety monitor with SLAM
3. Create RViz configuration
4. Test in Gazebo simulation
```

**Test command:**
```bash
# Full demo test
docker-compose up demo-slam
```

### **Day 5: Documentation & Validation**

```bash
# Tasks:
1. Update README with working demo
2. Create video demonstration
3. Fix any remaining integration issues
4. Prepare for Week 2 LLM integration
```

## **📊 Success Criteria for Week 1**

- [ ] `docker-compose up demo-slam` works without errors
- [ ] Robot moves safely in Gazebo simulation
- [ ] SLAM generates map visible in RViz
- [ ] Safety monitor prevents collisions
- [ ] Emergency stop works reliably (<100ms)
- [ ] All safety tests pass in CI/CD

## **🔧 Required Files Still Missing**

### **Build Files:**
```bash
Service/intelligence/eip_interfaces/CMakeLists.txt  ✅ Created
Service/core/eip_slam/CMakeLists.txt              ❌ Needed
Service/core/eip_slam/setup.py                    ❌ Needed  
Service/core/eip_slam/config/basic_slam.rviz      ❌ Needed
```

### **Integration Files:**
```bash
Service/integration/eip_orchestrator/launch/      ❌ Referenced in docker-compose
Service/benchmarks/safety_benchmarks/test_*.py   ⚠️  Only emergency_stop created
```

## **⚠️ Critical Issues to Address**

### **1. Package.xml Name Tags**
Current files have `<n>` instead of `<name>` - will cause build failures.

### **2. Python Executable Permissions**
All Python nodes need executable permissions and shebang lines.

### **3. ROS 2 Entry Points** 
Need setup.py files to register ROS 2 node executables.

### **4. Docker Display Setup**
X11 forwarding may need configuration for RViz.

## **🚧 Known Limitations for Week 1**

### **Acceptable Limitations:**
- No LLM integration (planned for Week 2)
- No real human detection (simulated for now)
- No object detection (SLAM mapping only)
- No continuous learning (basic operation only)

### **Must Work:**
- Safety emergency stop
- Basic SLAM mapping
- RViz visualization  
- Teleop control
- Docker environment

## **📞 Support Channels**

If you encounter issues:
1. Check build logs: `colcon build --event-handlers console_direct+`
2. Test individual components: `ros2 run <package> <executable>`
3. Validate ROS 2 setup: `ros2 doctor`
4. Review container logs: `docker-compose logs <service>`

---

**Week 1 Focus: Foundation Safety + Basic SLAM**
**Week 2 Preview: LLM Integration + VLM Grounding** 