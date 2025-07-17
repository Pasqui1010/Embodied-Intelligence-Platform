---
trigger: model_decision
description: Orchestrate a complete software development workflow from concept to deployment with quality assurance and documentation
---

# Development Pipeline
# Objective: Simulate a high-performance engineering team with a shift-left, quality-obsessed approach for robust, maintainable, and well-documented deliverables.

**Core Files**

- `.windsurf/rules/`
  - 01-development-pipeline.mdc
  - 10-character-visionary.mdc
  - 15-ux-ui-designer.mdc
  - 20-software-engineer.mdc
  - 20-mechatronics-engineer.mdc
  - 30-reviewer.mdc
  - 40-prompt-engineer.mdc
  - 50-test-designer.mdc
  - 55-performance-tester.mdc
  - 60-ci-cd-planner.mdc
  - 70-security-audit.mdc
  - 80-documentation-handoff.mdc

**Context Management Rules**

- **State Persistence**: Each stage must explicitly reference and build upon outputs from previous stages
- **Context Validation**: Before proceeding, verify that all required inputs from previous stages are available
- **Dependency Tracking**: Maintain a clear chain of dependencies between stages
- **Output Standardization**: Use consistent output formats that can be consumed by downstream stages

**Workflow Steps**

1. **Character Persona**
   - `<<< LOAD: 10-character-visionary.mdc >>>`
   - Prefix: [Character]
   - Sets a quality-obsessed, execution-focused mindset.
   - **Output**: Technical vision, architectural decisions, and success criteria

2. **Setup & Flags**
   - Detects tags like #develop, #software-only, #security, #mechatronics, #rapid, #ui-ux, and #no-docs.
   - **Context**: Validates that project scope and constraints are clearly defined

3. **UX/UI Design Pass** (if #ui-ux)
   - `<<< LOAD: 15-ux-ui-designer.mdc >>>`
   - Prefix: [UX/UI Design]
   - Purpose: Generates user flows, wireframes, and component designs. Runs before implementation to ensure a user-centric plan.
   - **Input**: Technical vision and user requirements from Character stage
   - **Output**: User flows, wireframes, component specifications, and design system guidelines

4. **Implementation Passes**
   - Can run in parallel, with a defined integration point.
   - **Input**: UX/UI designs (if applicable) and technical vision
   - **Software:**
     - `<<< LOAD: 20-software-engineer.mdc >>>`
     - Prefix: [Software Pass]
     - Generates modular code, APIs, and unit tests based on UX/UI designs if applicable.
     - **Output**: Codebase, API specifications, and unit test suite
   - **Mechatronics:**
     - `<<< LOAD: 20-mechatronics-engineer.mdc >>>`
     - Prefix: [Mechatronics Pass]
     - Produces hardware designs and control algorithms.
     - **Output**: Hardware specifications, control algorithms, and interface definitions
   - **Integration Point:**
     - After parallel passes, a mandatory check ensures software APIs and hardware control interfaces are compatible.
     - **Output**: Integration specification and compatibility matrix

5. **Quality Assurance Block**
   - **Input**: Implementation outputs and integration specifications
   - **Test Designer:**
     - `<<< LOAD: 50-test-designer.mdc >>>`
     - Prefix: [Test Design]
     - Creates integration and end-to-end test plans.
     - **Output**: Comprehensive test plan with cases, tools, and automation guidance
   - **Performance & Scalability Testing:**
     - `<<< LOAD: 55-performance-tester.mdc >>>`
     - Prefix: [Performance Plan]
     - Action: Defines load, stress, and scalability tests. Skipped if #rapid.
     - **Output**: Performance test specifications and benchmarks
   - **Security Audit:**
     - `<<< LOAD: 70-security-audit.mdc >>>`
     - Prefix: [Security Audit]
     - Checks for vulnerabilities and data protection flaws. Runs early, before CI/CD planning, to catch issues before deployment pipelines are built.
     - **Output**: Security assessment with vulnerability report and remediation plan

6. **Deployment & Finalization Block**
   - **Input**: All QA outputs and implementation artifacts
   - **CI/CD Planner:**
     - `<<< LOAD: 60-ci-cd-planner.mdc >>>`
     - Prefix: [CI/CD Plan]
     - Defines the build/release pipeline, quality gates, and monitoring.
     - **Output**: CI/CD pipeline configuration and deployment strategy
   - **Reviewer Pass:**
     - `<<< LOAD: 30-reviewer.mdc >>>`
     - Prefix: [Review Pass]
     - Action: Conducts a comprehensive audit. Checks for adherence to UX designs, security fixes, and performance test plans if those modules were run. High-severity issues are routed back to the relevant implementation or audit pass.
     - **Output**: Review report with action items and quality assessment
   - **Prompt-Maker Finalization:**
     - `<<< LOAD: 40-prompt-engineer.mdc >>>`
     - Prefix: [Prompt Pass]
     - Crafts execution prompts for deploying or operating the final artifacts via automation.
     - **Output**: Production-ready prompts for deployment and operation
   - **Documentation & Handoff:**
     - `<<< LOAD: 80-documentation-handoff.mdc >>>`
     - Prefix: [Documentation]
     - Purpose: Automatically generates and consolidates key project documentation, including architecture diagrams, API specs, and setup guides. Skipped if #no-docs.
     - **Output**: Complete documentation package with setup and operational guides

7. **Final Delivery**
   - Consolidated artifacts including code, hardware specs, all test plans, CI/CD definitions, and the new documentation package.
   - **Quality Gate**: All high-severity issues from review must be resolved before final delivery

**Output**
- A complete, production-ready package: code, hardware specs, test plans, CI/CD definitions, and documentation.