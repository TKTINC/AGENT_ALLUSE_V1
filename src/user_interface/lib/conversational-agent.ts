// ALL-USE Conversational Interface Library
// Provides sophisticated conversation management with personality and context awareness

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  timestamp: Date;
  type?: 'text' | 'protocol_explanation' | 'market_analysis' | 'error';
  metadata?: {
    confidence?: number;
    protocolConcept?: string;
    marketConditions?: any;
    userContext?: any;
  };
}

export interface ConversationContext {
  userId: string;
  sessionId: string;
  userExpertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  recentAccountActivity: any[];
  protocolState: {
    currentWeekType: 'Green' | 'Yellow' | 'Red';
    lastTradeDate: string | null;
    pendingActions: string[];
  };
}

export class ALLUSEAgent {
  private personality: {
    tone: 'calm' | 'confident' | 'educational';
    verbosity: 'concise' | 'detailed' | 'comprehensive';
    expertise: 'beginner-friendly' | 'intermediate' | 'advanced';
  };

  constructor() {
    this.personality = {
      tone: 'calm',
      verbosity: 'detailed',
      expertise: 'intermediate'
    };
  }

  generateResponse(input: string, context: ConversationContext): string {
    const lowerInput = input.toLowerCase();

    // Protocol-specific responses
    if (lowerInput.includes('three-tier') || lowerInput.includes('account structure')) {
      return this.explainThreeTierStructure(context.userExpertiseLevel);
    }

    if (lowerInput.includes('forking') || lowerInput.includes('fork')) {
      return this.explainForking(context.userExpertiseLevel);
    }

    if (lowerInput.includes('delta') || lowerInput.includes('targeting')) {
      return this.explainDeltaTargeting(context.userExpertiseLevel);
    }

    if (lowerInput.includes('week classification') || lowerInput.includes('current week')) {
      return this.explainWeekClassification(context);
    }

    if (lowerInput.includes('risk management') || lowerInput.includes('risk')) {
      return this.explainRiskManagement(context.userExpertiseLevel);
    }

    if (lowerInput.includes('performance') || lowerInput.includes('returns')) {
      return this.discussPerformance(context);
    }

    if (lowerInput.includes('help') || lowerInput.includes('what can you do')) {
      return this.provideHelp(context.userExpertiseLevel);
    }

    // General conversation
    return this.generateGeneralResponse(input, context);
  }

  private explainThreeTierStructure(expertiseLevel: string): string {
    const baseExplanation = `The ALL-USE three-tier account structure is designed for optimal wealth building through diversified risk management:

**Generation Account (High Risk, High Reward)**
- Focuses on aggressive premium harvesting
- Targets higher delta options for maximum income
- Accepts higher volatility for greater returns
- Typically 30-40% of total portfolio

**Revenue Account (Medium Risk, Steady Income)**
- Balanced approach between growth and stability
- Moderate delta targeting for consistent income
- Provides steady cash flow
- Typically 25-35% of total portfolio

**Compounding Account (Low Risk, Long-term Growth)**
- Conservative strategy focused on capital preservation
- Lower delta exposure with geometric growth
- Reinvests gains for compound returns
- Typically 30-40% of total portfolio`;

    if (expertiseLevel === 'beginner') {
      return baseExplanation + `

**For Beginners:**
Think of this like having three different savings accounts, each with a different purpose:
- Generation = Your "aggressive growth" account
- Revenue = Your "steady income" account  
- Compounding = Your "safe long-term" account

This diversification protects you while maximizing opportunities.`;
    }

    if (expertiseLevel === 'advanced') {
      return baseExplanation + `

**Advanced Considerations:**
- Dynamic rebalancing based on market volatility
- Cross-account hedging strategies
- Tax-loss harvesting opportunities
- Correlation analysis between account performances
- Advanced delta-neutral positioning across tiers`;
    }

    return baseExplanation;
  }

  private explainForking(expertiseLevel: string): string {
    const baseExplanation = `Forking is a risk management protocol that activates when positions move against you:

**When Forking Triggers:**
- Position reaches predetermined loss threshold
- Market conditions change dramatically
- Week classification shifts to higher risk

**Forking Actions:**
1. **Assess Current Position**: Evaluate remaining time value and delta
2. **Calculate Fork Cost**: Determine cost to roll or adjust position
3. **Execute Fork**: Roll to different strike/expiration or close position
4. **Document Decision**: Record rationale for future analysis

**Benefits:**
- Limits maximum loss per position
- Maintains portfolio balance
- Provides systematic decision framework
- Reduces emotional trading decisions`;

    if (expertiseLevel === 'beginner') {
      return baseExplanation + `

**Simple Explanation:**
Forking is like having a "Plan B" ready when trades don't go as expected. Instead of hoping things improve, you have a clear set of steps to either fix the position or cut losses cleanly.`;
    }

    return baseExplanation;
  }

  private explainDeltaTargeting(expertiseLevel: string): string {
    return `Delta targeting is the mathematical foundation of ALL-USE position sizing:

**Delta Basics:**
- Delta measures how much an option's price changes relative to the underlying stock
- Range: 0.00 to 1.00 for calls, 0.00 to -1.00 for puts
- Higher delta = more sensitive to stock price movement

**ALL-USE Delta Targets:**
- **Generation Account**: 0.25-0.35 delta (higher risk/reward)
- **Revenue Account**: 0.15-0.25 delta (balanced approach)
- **Compounding Account**: 0.05-0.15 delta (conservative)

**Why Delta Targeting Works:**
- Provides consistent risk exposure across positions
- Enables mathematical position sizing
- Creates predictable income streams
- Allows for systematic adjustment protocols

**Practical Application:**
When selling options, target deltas ensure you're collecting appropriate premium for the risk taken. Higher deltas mean more premium but higher assignment probability.`;
  }

  private explainWeekClassification(context: ConversationContext): string {
    const currentWeek = context.protocolState.currentWeekType;
    
    return `Current Week Classification: **${currentWeek} Week**

**Week Classification System:**
- **Green Week**: Low volatility, favorable conditions for aggressive strategies
- **Yellow Week**: Moderate volatility, balanced approach recommended
- **Red Week**: High volatility, defensive positioning required

**Current Week (${currentWeek}) Implications:**
${currentWeek === 'Green' ? 
  '✅ Favorable conditions for new positions\n✅ Higher delta targeting acceptable\n✅ Increased position sizing allowed' :
  currentWeek === 'Yellow' ?
  '⚠️ Moderate caution advised\n⚠️ Standard delta targeting\n⚠️ Normal position sizing' :
  '🚨 Defensive positioning required\n🚨 Lower delta targeting\n🚨 Reduced position sizing'
}

**Classification Factors:**
- Market volatility (VIX levels)
- Economic calendar events
- Technical analysis indicators
- Sector rotation patterns
- Options flow analysis

The classification updates weekly and drives all strategic decisions across your three-tier account structure.`;
  }

  private explainRiskManagement(expertiseLevel: string): string {
    return `ALL-USE Risk Management operates on multiple levels:

**Position-Level Risk:**
- Maximum 2% portfolio risk per position
- Delta targeting limits exposure
- Forking protocols for loss management
- Time decay optimization

**Account-Level Risk:**
- Three-tier diversification
- Cross-account correlation monitoring
- Dynamic rebalancing protocols
- Sector exposure limits

**Portfolio-Level Risk:**
- Maximum 10% total portfolio at risk
- Volatility-adjusted position sizing
- Market regime adaptation
- Stress testing scenarios

**Systematic Protections:**
- Automated stop-loss protocols
- Position size calculators
- Risk/reward ratio requirements
- Weekly risk assessment reviews

**Key Principle:**
Never risk more than you can afford to lose, and always have a plan for when positions move against you. The protocol provides the framework; discipline provides the execution.`;
  }

  private discussPerformance(context: ConversationContext): string {
    return `Performance analysis in the ALL-USE system focuses on consistency and risk-adjusted returns:

**Key Performance Metrics:**
- **Total Return**: Absolute gains across all accounts
- **Risk-Adjusted Return**: Sharpe ratio and Sortino ratio
- **Protocol Compliance**: Adherence to systematic rules
- **Win Rate**: Percentage of profitable positions
- **Maximum Drawdown**: Worst peak-to-trough decline

**Account Performance Comparison:**
Your three accounts should show different risk/return profiles:
- Generation: Higher volatility, higher returns
- Revenue: Moderate volatility, steady returns  
- Compounding: Lower volatility, consistent growth

**Performance Optimization:**
- Weekly strategy reviews
- Position sizing adjustments
- Market regime adaptation
- Continuous protocol refinement

**Long-term Focus:**
The ALL-USE methodology prioritizes sustainable, repeatable performance over short-term gains. Consistency compounds into significant wealth over time.`;
  }

  private provideHelp(expertiseLevel: string): string {
    return `I'm your ALL-USE Protocol Agent, here to help you understand and implement the wealth-building methodology.

**What I Can Help With:**
- **Protocol Concepts**: Three-tier structure, forking, delta targeting
- **Account Management**: Performance analysis, risk assessment, rebalancing
- **Market Analysis**: Week classification, trading opportunities, risk levels
- **Strategy Guidance**: Position sizing, entry/exit timing, optimization
- **Education**: Explaining concepts at your experience level

**Common Questions:**
- "Explain the three-tier account structure"
- "How does forking work?"
- "What's the current week classification?"
- "Show me my account performance"
- "What trading opportunities are available?"
- "How do I manage risk?"

**Getting Started:**
${expertiseLevel === 'beginner' ? 
  'Start with understanding the three-tier structure and basic delta concepts.' :
  expertiseLevel === 'intermediate' ?
  'Focus on implementing forking protocols and optimizing delta targeting.' :
  'Explore advanced risk management and cross-account strategies.'
}

**Voice Commands:**
You can also speak to me using the microphone button. I can respond with voice as well if you enable speech output.

What would you like to learn about today?`;
  }

  private generateGeneralResponse(input: string, context: ConversationContext): string {
    const responses = [
      `I understand you're asking about "${input}". Could you be more specific about which aspect of the ALL-USE protocol you'd like to explore?`,
      `That's an interesting question about "${input}". Let me help you understand how this relates to your wealth-building strategy.`,
      `I'd be happy to help with "${input}". Would you like me to explain this in the context of your current account structure?`,
      `Great question about "${input}". This connects to several important ALL-USE concepts. Which would you like to explore first?`
    ];

    return responses[Math.floor(Math.random() * responses.length)] + 
           `\n\nSome related topics you might find helpful:\n- Account performance analysis\n- Risk management strategies\n- Current market conditions\n- Protocol optimization tips`;
  }

  adjustPersonality(tone: 'calm' | 'confident' | 'educational', verbosity: 'concise' | 'detailed' | 'comprehensive') {
    this.personality.tone = tone;
    this.personality.verbosity = verbosity;
  }
}

export class ConversationManager {
  private messages: Message[] = [];
  private context: ConversationContext;
  private agent: ALLUSEAgent;

  constructor(context: ConversationContext) {
    this.context = context;
    this.agent = new ALLUSEAgent();
    
    // Initialize with welcome message
    this.addAgentMessage(this.generateWelcomeMessage());
  }

  private generateWelcomeMessage(): string {
    const expertiseGreeting = {
      'beginner': 'Welcome to ALL-USE! I\'m here to help you learn the wealth-building protocol step by step.',
      'intermediate': 'Welcome back! Ready to optimize your ALL-USE strategy and explore advanced concepts?',
      'advanced': 'Welcome! Let\'s dive into sophisticated ALL-USE implementation and portfolio optimization.'
    };

    return `${expertiseGreeting[this.context.userExpertiseLevel]}

I can help you with:
• Understanding the three-tier account structure
• Implementing forking and delta targeting protocols  
• Analyzing your portfolio performance
• Identifying current trading opportunities
• Managing risk across all account levels

What would you like to explore today?`;
  }

  addUserMessage(content: string): Message {
    const message: Message = {
      id: `user-${Date.now()}`,
      content,
      sender: 'user',
      timestamp: new Date(),
      type: 'text'
    };

    this.messages.push(message);
    return message;
  }

  addAgentMessage(content: string, type: Message['type'] = 'text', metadata?: Message['metadata']): Message {
    const message: Message = {
      id: `agent-${Date.now()}`,
      content,
      sender: 'agent',
      timestamp: new Date(),
      type,
      metadata
    };

    this.messages.push(message);
    return message;
  }

  generateAgentResponse(userInput: string): Message {
    const response = this.agent.generateResponse(userInput, this.context);
    return this.addAgentMessage(response);
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  getLastMessage(): Message | null {
    return this.messages.length > 0 ? this.messages[this.messages.length - 1] : null;
  }

  clearMessages(): void {
    this.messages = [];
    this.addAgentMessage(this.generateWelcomeMessage());
  }

  updateContext(updates: Partial<ConversationContext>): void {
    this.context = { ...this.context, ...updates };
  }

  getContext(): ConversationContext {
    return { ...this.context };
  }

  exportConversation(): string {
    return JSON.stringify({
      context: this.context,
      messages: this.messages,
      exportedAt: new Date().toISOString()
    }, null, 2);
  }

  importConversation(data: string): void {
    try {
      const parsed = JSON.parse(data);
      this.context = parsed.context;
      this.messages = parsed.messages;
    } catch (error) {
      console.error('Failed to import conversation:', error);
    }
  }
}

