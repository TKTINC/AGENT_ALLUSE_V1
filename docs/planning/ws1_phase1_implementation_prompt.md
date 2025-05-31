# ALL-USE Agent: WS1-P1 Implementation Prompt

## 1. Introduction

This document serves as the implementation prompt for Workstream 1, Phase 1 (Agent Foundation) of the ALL-USE agent development. Following the iAI framework, this prompt defines the objectives, requirements, and approach for implementing the core agent architecture that will serve as the foundation for all other workstreams.

## 2. Objectives

The primary objectives for WS1-P1 implementation are:

1. Establish the core agent architecture with perception-cognition-action loop
2. Implement basic memory systems for conversation and protocol state
3. Create a simple cognitive framework for decision-making
4. Develop foundational personality and dialogue capabilities

## 3. Requirements

### 3.1 Core Agent Architecture

- Design and implement a modular agent architecture with clear separation of concerns
- Create a perception-cognition-action loop that processes inputs, makes decisions, and generates outputs
- Implement event handling and processing pipeline
- Establish communication channels between components
- Create a plugin system for extending functionality in later phases

### 3.2 Memory Systems

- Implement a basic conversation memory system that stores:
  - User inputs and agent responses
  - Conversation context and history
  - User preferences and settings
- Create a protocol state memory system that maintains:
  - Current account structure state
  - Trading protocol state
  - Decision history and rationale
- Implement simple retrieval mechanisms for both memory systems

### 3.3 Cognitive Framework

- Implement a basic decision-making framework that:
  - Processes inputs from perception module
  - Applies simple reasoning based on protocol rules
  - Generates decisions for action module
- Create a context management system that maintains awareness of:
  - Current conversation state
  - User goals and preferences
  - System capabilities and limitations
- Implement basic error handling and recovery mechanisms

### 3.4 Personality and Dialogue

- Implement foundational personality traits as defined in the specification:
  - Methodical
  - Confident
  - Educational
  - Calm
  - Precise
- Create basic dialogue capabilities for:
  - Greeting and introduction
  - Protocol explanation
  - Simple question answering
  - Error handling and clarification
- Implement a template-based response generation system

## 4. Technical Approach

### 4.1 Architecture Design

- Use a modular, component-based architecture
- Implement clear interfaces between components
- Use dependency injection for component communication
- Create a plugin system for extensibility
- Implement event-driven communication where appropriate

### 4.2 Implementation Strategy

- Use Python as the primary implementation language
- Implement components as classes with clear responsibilities
- Use type hints for better code quality and documentation
- Create comprehensive unit tests for all components
- Implement integration tests for component interactions

### 4.3 Testing Approach

- Unit tests for all components and functions
- Integration tests for component interactions
- System tests for end-to-end functionality
- Test coverage target: 80% minimum

## 5. Integration Points

### 5.1 Integration with Other Workstreams

- Define clear interfaces for integration with:
  - WS2: Protocol Engine
  - WS3: Account Management
  - WS4: Market Integration
  - WS5: Learning System
  - WS6: User Interface

### 5.2 External Dependencies

- Identify and document external dependencies
- Implement abstraction layers for external services
- Create mock implementations for testing

## 6. Deliverables

The following deliverables are expected for WS1-P1:

1. Core agent architecture implementation
2. Basic memory systems for conversation and protocol state
3. Simple cognitive framework for decision-making
4. Foundational personality and dialogue capabilities
5. Comprehensive test suite
6. Technical documentation

## 7. Implementation Timeline

WS1-P1 implementation is scheduled for Weeks 1-2:

- **Week 1**
  - Core agent architecture design and implementation
  - Basic memory systems implementation
  - Initial test suite development

- **Week 2**
  - Simple cognitive framework implementation
  - Foundational personality and dialogue capabilities
  - Comprehensive testing and documentation

## 8. Success Criteria

WS1-P1 implementation will be considered successful when:

1. All required components are implemented according to specifications
2. The agent can process inputs, make simple decisions, and generate outputs
3. Basic memory systems successfully store and retrieve information
4. The agent demonstrates the defined personality traits in interactions
5. All tests pass and meet the coverage requirements
6. Documentation is complete and accurate

## 9. Next Steps

Upon completion of WS1-P1, the implementation response document will be created to detail the actual implementation, any deviations from the plan, challenges encountered, and lessons learned. This will be followed by a review before proceeding to WS2-P1 implementation.

---

This implementation prompt serves as the guiding document for WS1-P1 of the ALL-USE agent development. It ensures alignment with the iAI framework by clearly defining the requirements, approach, and deliverables before implementation begins.
