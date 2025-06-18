import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, Send, Mic, MicOff, Volume2, VolumeX, User, Bot, Lightbulb, TrendingUp } from 'lucide-react';
import { ALLUSEAgent, ConversationManager, Message, ConversationContext } from '../lib/conversational-agent';
import { ws1Agent, ProtocolExplanation, WeekClassification, TradingOpportunity } from '../lib/ws1-integration';

interface ConversationalInterfaceProps {
  className?: string;
  onMessageSent?: (message: Message) => void;
  onAgentResponse?: (message: Message) => void;
}

export const ConversationalInterface: React.FC<ConversationalInterfaceProps> = ({
  className = '',
  onMessageSent,
  onAgentResponse
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [conversationManager, setConversationManager] = useState<ConversationManager | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  // Initialize conversation manager and WS1 integration
  useEffect(() => {
    const initializeServices = async () => {
      // Initialize WS1 Agent Foundation connection
      try {
        await ws1Agent.connect();
        console.log('WS1 Agent Foundation connected successfully');
      } catch (error) {
        console.error('Failed to connect to WS1 Agent Foundation:', error);
      }

      // Initialize conversation manager
      const context: ConversationContext = {
        userId: 'demo-user',
        sessionId: Date.now().toString(),
        userExpertiseLevel: 'intermediate',
        recentAccountActivity: [],
        protocolState: {
          currentWeekType: 'Green',
          lastTradeDate: null,
          pendingActions: []
        }
      };

      const manager = new ConversationManager(context);
      setConversationManager(manager);
      setMessages(manager.getMessages());
    };

    initializeServices();

    // Cleanup on unmount
    return () => {
      if (ws1Agent.isConnectionActive()) {
        ws1Agent.disconnect();
      }
    };
  }, []);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage(transcript);
        setIsListening(false);
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }

    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !conversationManager) return;

    const userMessage = conversationManager.addUserMessage(inputMessage);
    setMessages([...conversationManager.getMessages()]);
    onMessageSent?.(userMessage);
    
    const currentInput = inputMessage;
    setInputMessage('');
    setIsLoading(true);
    setShowSuggestions(false);

    try {
      // Enhanced response generation with WS1 integration
      let agentResponse: Message;

      // Check if the message is asking for specific protocol information
      if (await isProtocolQuestion(currentInput)) {
        agentResponse = await generateEnhancedResponse(currentInput, conversationManager);
      } else {
        // Use standard conversation manager for general queries
        agentResponse = conversationManager.generateAgentResponse(currentInput);
      }

      setMessages([...conversationManager.getMessages()]);
      setIsLoading(false);
      onAgentResponse?.(agentResponse);

      // Auto-speak response if speech is enabled
      if (isSpeaking && synthRef.current) {
        speakMessage(agentResponse.content);
      }
    } catch (error) {
      console.error('Error generating response:', error);
      const errorResponse = conversationManager.addAgentMessage(
        "I apologize, but I'm experiencing some technical difficulties. Please try again or rephrase your question."
      );
      setMessages([...conversationManager.getMessages()]);
      setIsLoading(false);
    }
  };

  // Enhanced response generation using WS1 integration
  const generateEnhancedResponse = async (input: string, manager: ConversationManager): Promise<Message> => {
    const lowerInput = input.toLowerCase();

    try {
      // Protocol concept explanations
      if (lowerInput.includes('three-tier') || lowerInput.includes('account structure')) {
        const explanation = await ws1Agent.explainProtocolConcept('three-tier-account-structure');
        return manager.addAgentMessage(formatProtocolExplanation(explanation));
      }

      if (lowerInput.includes('forking') || lowerInput.includes('fork')) {
        const explanation = await ws1Agent.explainProtocolConcept('forking-protocol');
        return manager.addAgentMessage(formatProtocolExplanation(explanation));
      }

      if (lowerInput.includes('week classification') || lowerInput.includes('current week')) {
        const classification = await ws1Agent.getCurrentWeekClassification();
        return manager.addAgentMessage(formatWeekClassification(classification));
      }

      if (lowerInput.includes('trading opportunities') || lowerInput.includes('opportunities')) {
        const opportunities = await ws1Agent.getTradingOpportunities();
        return manager.addAgentMessage(formatTradingOpportunities(opportunities));
      }

      if (lowerInput.includes('delta') || lowerInput.includes('targeting')) {
        const explanation = await ws1Agent.explainProtocolConcept('delta-targeting');
        return manager.addAgentMessage(formatProtocolExplanation(explanation));
      }

      if (lowerInput.includes('risk management') || lowerInput.includes('risk')) {
        const explanation = await ws1Agent.explainProtocolConcept('risk-management');
        return manager.addAgentMessage(formatProtocolExplanation(explanation));
      }

      if (lowerInput.includes('market conditions') || lowerInput.includes('market analysis')) {
        const analysis = await ws1Agent.analyzeMarketConditions();
        return manager.addAgentMessage(formatMarketAnalysis(analysis));
      }

      // Default to standard response if no specific protocol question detected
      return manager.generateAgentResponse(input);
    } catch (error) {
      console.error('Error in enhanced response generation:', error);
      return manager.addAgentMessage(
        "I encountered an issue accessing the latest protocol information. Let me provide you with general guidance instead. " +
        manager.generateAgentResponse(input).content
      );
    }
  };

  // Helper function to detect protocol-related questions
  const isProtocolQuestion = async (input: string): Promise<boolean> => {
    const protocolKeywords = [
      'three-tier', 'account structure', 'forking', 'fork',
      'week classification', 'current week', 'trading opportunities',
      'delta', 'targeting', 'risk management', 'market conditions'
    ];

    return protocolKeywords.some(keyword => 
      input.toLowerCase().includes(keyword)
    );
  };

  // Formatting functions for WS1 responses
  const formatProtocolExplanation = (explanation: ProtocolExplanation): string => {
    let response = `## ${explanation.concept}\n\n`;
    response += `${explanation.explanation}\n\n`;
    
    if (explanation.examples.length > 0) {
      response += `**Examples:**\n`;
      explanation.examples.forEach((example, index) => {
        response += `${index + 1}. ${example}\n`;
      });
      response += '\n';
    }

    if (explanation.implementationSteps.length > 0) {
      response += `**Implementation Steps:**\n`;
      explanation.implementationSteps.forEach(step => {
        response += `• ${step}\n`;
      });
      response += '\n';
    }

    if (explanation.riskFactors.length > 0) {
      response += `**Risk Considerations:**\n`;
      explanation.riskFactors.forEach(risk => {
        response += `⚠️ ${risk}\n`;
      });
    }

    return response;
  };

  const formatWeekClassification = (classification: WeekClassification): string => {
    let response = `## Current Week Classification: ${classification.classification.toUpperCase()}\n\n`;
    response += `**Week ${classification.currentWeek}** - Confidence: ${(classification.confidence * 100).toFixed(0)}%\n\n`;
    response += `**Analysis:** ${classification.reasoning}\n\n`;
    
    response += `**Market Conditions:**\n`;
    response += `• Volatility: ${classification.marketConditions.volatility}%\n`;
    response += `• Trend: ${classification.marketConditions.trend}\n`;
    response += `• Volume: ${classification.marketConditions.volume}\n`;
    response += `• Sentiment: ${classification.marketConditions.sentiment}\n\n`;

    if (classification.tradingRecommendations.length > 0) {
      response += `**Trading Recommendations:**\n`;
      classification.tradingRecommendations.forEach((rec, index) => {
        response += `${index + 1}. **${rec.action}** (${rec.riskLevel} risk)\n`;
        response += `   ${rec.rationale}\n`;
        response += `   Timeframe: ${rec.timeframe}\n\n`;
      });
    }

    return response;
  };

  const formatTradingOpportunities = (opportunities: TradingOpportunity[]): string => {
    let response = `## Current Trading Opportunities\n\n`;
    response += `Found ${opportunities.length} opportunities across all account tiers:\n\n`;

    opportunities.forEach((opp, index) => {
      response += `**${index + 1}. ${opp.symbol} - ${opp.strategy}** (${opp.accountType.toUpperCase()})\n`;
      response += `• Entry: $${opp.entryPrice} | Target: $${opp.targetPrice} | Stop: $${opp.stopLoss}\n`;
      response += `• Probability: ${(opp.probability * 100).toFixed(0)}% | Risk/Reward: ${opp.riskReward}:1\n`;
      response += `• Delta Target: ${opp.deltaTarget} | Expires: ${opp.expirationDate}\n`;
      response += `• Reasoning: ${opp.reasoning}\n\n`;
    });

    response += `*These opportunities are based on current market analysis and should be validated before execution.*`;
    return response;
  };

  const formatMarketAnalysis = (analysis: any): string => {
    let response = `## Market Analysis\n\n`;
    response += `**Overall Assessment:** ${analysis.overall}\n`;
    response += `**Current Volatility:** ${analysis.volatility}%\n`;
    response += `**Market Trend:** ${analysis.trend}\n`;
    response += `**Sentiment:** ${analysis.sentiment}\n\n`;

    if (analysis.recommendations.length > 0) {
      response += `**Recommendations:**\n`;
      analysis.recommendations.forEach((rec: string, index: number) => {
        response += `${index + 1}. ${rec}\n`;
      });
    }

    return response;
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleSpeechRecognition = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const toggleSpeechSynthesis = () => {
    setIsSpeaking(!isSpeaking);
    if (isSpeaking && synthRef.current) {
      synthRef.current.cancel();
    }
  };

  const speakMessage = (text: string) => {
    if (!synthRef.current) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 0.8;
    
    synthRef.current.speak(utterance);
  };

  const handleSuggestedQuestion = (question: string) => {
    setInputMessage(question);
    setShowSuggestions(false);
  };

  const suggestedQuestions = [
    "Explain the three-tier account structure",
    "How does the forking protocol work?",
    "What's the current week classification?",
    "Show me my account performance",
    "What trading opportunities are available?",
    "Explain the risk management protocols"
  ];

  return (
    <div className={`flex flex-col h-full bg-white ${className}`}>
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 flex items-center gap-3">
        <MessageCircle className="w-6 h-6" />
        <h2 className="text-xl font-semibold">ALL-USE Protocol Agent</h2>
        <div className="ml-auto flex gap-2">
          <button
            onClick={toggleSpeechRecognition}
            className={`p-2 rounded-lg transition-colors ${
              isListening ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-400'
            }`}
            title={isListening ? 'Stop voice input' : 'Start voice input'}
          >
            {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>
          <button
            onClick={toggleSpeechSynthesis}
            className={`p-2 rounded-lg transition-colors ${
              isSpeaking ? 'bg-green-500 hover:bg-green-600' : 'bg-blue-500 hover:bg-blue-400'
            }`}
            title={isSpeaking ? 'Disable speech' : 'Enable speech'}
          >
            {isSpeaking ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.sender === 'agent' && (
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
            )}
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-gray-100 text-gray-900'
                  : 'bg-blue-600 text-white'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className="text-xs opacity-70 mt-1">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
            {message.sender === 'user' && (
              <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center flex-shrink-0">
                <User className="w-5 h-5 text-white" />
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="flex gap-3 justify-start">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="bg-blue-600 text-white p-3 rounded-lg">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {showSuggestions && messages.length <= 1 && (
        <div className="p-4 border-t bg-gray-50">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-gray-700">Suggested Questions</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSuggestedQuestion(question)}
                className="text-left p-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t bg-white">
        <div className="flex gap-2">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about the ALL-USE protocol, accounts, or trading decisions..."
            className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            <Send className="w-4 h-4" />
            Send
          </button>
        </div>
        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <div className="flex items-center gap-4">
            <span>🎤 Voice input available</span>
            <span>🔊 Speech output {isSpeaking ? 'enabled' : 'disabled'}</span>
          </div>
          <span>Press Enter to send, Shift+Enter for new line</span>
        </div>
      </div>
    </div>
  );
};

