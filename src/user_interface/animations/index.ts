// Advanced Animation System for WS6-P3
// Professional animations, micro-interactions, and visual feedback

import { motion, AnimatePresence, useSpring, useMotionValue, useTransform, Variants } from 'framer-motion';
import { useCallback, useEffect, useRef, useState } from 'react';

// Animation Presets and Configurations
export const animationPresets = {
  // Page transitions
  pageTransition: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.3, ease: 'easeInOut' }
  },

  // Modal animations
  modalOverlay: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
    transition: { duration: 0.2 }
  },

  modalContent: {
    initial: { opacity: 0, scale: 0.9, y: 20 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exit: { opacity: 0, scale: 0.9, y: 20 },
    transition: { duration: 0.3, ease: 'easeOut' }
  },

  // Card animations
  cardHover: {
    whileHover: { 
      scale: 1.02, 
      y: -2,
      boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
      transition: { duration: 0.2 }
    },
    whileTap: { scale: 0.98 }
  },

  // Button animations
  buttonPress: {
    whileHover: { scale: 1.05 },
    whileTap: { scale: 0.95 },
    transition: { duration: 0.1 }
  },

  // List item animations
  listItem: {
    initial: { opacity: 0, x: -20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 },
    transition: { duration: 0.2 }
  },

  // Stagger animations
  staggerContainer: {
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  },

  // Loading animations
  pulse: {
    animate: {
      scale: [1, 1.05, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    }
  },

  // Success/Error feedback
  successFeedback: {
    initial: { scale: 0, opacity: 0 },
    animate: { 
      scale: [0, 1.2, 1], 
      opacity: 1,
      transition: { duration: 0.5, ease: 'easeOut' }
    },
    exit: { scale: 0, opacity: 0 }
  }
};

// Advanced Animation Hooks
export const useAdvancedAnimation = () => {
  const [isAnimating, setIsAnimating] = useState(false);
  const animationQueue = useRef<Array<() => Promise<void>>>([]);

  const queueAnimation = useCallback((animation: () => Promise<void>) => {
    animationQueue.current.push(animation);
    if (!isAnimating) {
      processQueue();
    }
  }, [isAnimating]);

  const processQueue = useCallback(async () => {
    if (animationQueue.current.length === 0) return;
    
    setIsAnimating(true);
    
    while (animationQueue.current.length > 0) {
      const animation = animationQueue.current.shift();
      if (animation) {
        await animation();
      }
    }
    
    setIsAnimating(false);
  }, []);

  return { queueAnimation, isAnimating };
};

// Gesture Recognition System
export const useGestureRecognition = () => {
  const [gestures, setGestures] = useState<{
    swipeLeft: boolean;
    swipeRight: boolean;
    swipeUp: boolean;
    swipeDown: boolean;
    pinch: boolean;
    rotate: boolean;
  }>({
    swipeLeft: false,
    swipeRight: false,
    swipeUp: false,
    swipeDown: false,
    pinch: false,
    rotate: false
  });

  const touchStart = useRef<{ x: number; y: number; time: number } | null>(null);
  const touchEnd = useRef<{ x: number; y: number; time: number } | null>(null);

  const handleTouchStart = useCallback((event: TouchEvent) => {
    const touch = event.touches[0];
    touchStart.current = {
      x: touch.clientX,
      y: touch.clientY,
      time: Date.now()
    };
  }, []);

  const handleTouchEnd = useCallback((event: TouchEvent) => {
    if (!touchStart.current) return;

    const touch = event.changedTouches[0];
    touchEnd.current = {
      x: touch.clientX,
      y: touch.clientY,
      time: Date.now()
    };

    const deltaX = touchEnd.current.x - touchStart.current.x;
    const deltaY = touchEnd.current.y - touchStart.current.y;
    const deltaTime = touchEnd.current.time - touchStart.current.time;

    // Minimum swipe distance and maximum time
    const minSwipeDistance = 50;
    const maxSwipeTime = 300;

    if (deltaTime > maxSwipeTime) return;

    const absDeltaX = Math.abs(deltaX);
    const absDeltaY = Math.abs(deltaY);

    if (absDeltaX > minSwipeDistance && absDeltaX > absDeltaY) {
      // Horizontal swipe
      if (deltaX > 0) {
        setGestures(prev => ({ ...prev, swipeRight: true }));
        setTimeout(() => setGestures(prev => ({ ...prev, swipeRight: false })), 100);
      } else {
        setGestures(prev => ({ ...prev, swipeLeft: true }));
        setTimeout(() => setGestures(prev => ({ ...prev, swipeLeft: false })), 100);
      }
    } else if (absDeltaY > minSwipeDistance && absDeltaY > absDeltaX) {
      // Vertical swipe
      if (deltaY > 0) {
        setGestures(prev => ({ ...prev, swipeDown: true }));
        setTimeout(() => setGestures(prev => ({ ...prev, swipeDown: false })), 100);
      } else {
        setGestures(prev => ({ ...prev, swipeUp: true }));
        setTimeout(() => setGestures(prev => ({ ...prev, swipeUp: false })), 100);
      }
    }

    touchStart.current = null;
    touchEnd.current = null;
  }, []);

  useEffect(() => {
    document.addEventListener('touchstart', handleTouchStart);
    document.addEventListener('touchend', handleTouchEnd);

    return () => {
      document.removeEventListener('touchstart', handleTouchStart);
      document.removeEventListener('touchend', handleTouchEnd);
    };
  }, [handleTouchStart, handleTouchEnd]);

  return gestures;
};

// Advanced Voice Interface Enhancement
export const useAdvancedVoiceInterface = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [isSupported, setIsSupported] = useState(false);
  const [commands, setCommands] = useState<Map<string, () => void>>(new Map());

  const recognition = useRef<any>(null);

  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      recognition.current = new SpeechRecognition();
      
      recognition.current.continuous = true;
      recognition.current.interimResults = true;
      recognition.current.lang = 'en-US';

      recognition.current.onstart = () => {
        setIsListening(true);
      };

      recognition.current.onend = () => {
        setIsListening(false);
      };

      recognition.current.onresult = (event: any) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          const confidence = event.results[i][0].confidence;

          if (event.results[i].isFinal) {
            finalTranscript += transcript;
            setConfidence(confidence);
            
            // Check for voice commands
            checkVoiceCommands(transcript.toLowerCase().trim());
          } else {
            interimTranscript += transcript;
          }
        }

        setTranscript(finalTranscript || interimTranscript);
      };

      recognition.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      setIsSupported(true);
    }
  }, []);

  const checkVoiceCommands = useCallback((transcript: string) => {
    for (const [command, action] of commands.entries()) {
      if (transcript.includes(command.toLowerCase())) {
        action();
        break;
      }
    }
  }, [commands]);

  const startListening = useCallback(() => {
    if (recognition.current && !isListening) {
      recognition.current.start();
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognition.current && isListening) {
      recognition.current.stop();
    }
  }, [isListening]);

  const addCommand = useCallback((command: string, action: () => void) => {
    setCommands(prev => new Map(prev.set(command, action)));
  }, []);

  const removeCommand = useCallback((command: string) => {
    setCommands(prev => {
      const newCommands = new Map(prev);
      newCommands.delete(command);
      return newCommands;
    });
  }, []);

  const speak = useCallback((text: string, options?: SpeechSynthesisUtteranceOptions) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      
      if (options) {
        Object.assign(utterance, options);
      }

      speechSynthesis.speak(utterance);
    }
  }, []);

  return {
    isListening,
    transcript,
    confidence,
    isSupported,
    startListening,
    stopListening,
    addCommand,
    removeCommand,
    speak
  };
};

// Professional Animation Components
export const AnimatedCard: React.FC<{
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  delay?: number;
}> = ({ children, className = '', onClick, delay = 0 }) => {
  return (
    <motion.div
      className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg ${className}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3, ease: 'easeOut' }}
      {...animationPresets.cardHover}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
};

export const AnimatedButton: React.FC<{
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'success' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
}> = ({ 
  children, 
  className = '', 
  onClick, 
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false
}) => {
  const baseClasses = 'rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2';
  
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600',
    success: 'bg-green-600 text-white hover:bg-green-700',
    danger: 'bg-red-600 text-white hover:bg-red-700'
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  return (
    <motion.button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className} ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      }`}
      onClick={onClick}
      disabled={disabled || loading}
      {...animationPresets.buttonPress}
    >
      {loading ? (
        <motion.div
          className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
      ) : null}
      {children}
    </motion.button>
  );
};

export const AnimatedList: React.FC<{
  children: React.ReactNode;
  className?: string;
  stagger?: boolean;
}> = ({ children, className = '', stagger = true }) => {
  return (
    <motion.div
      className={className}
      variants={stagger ? animationPresets.staggerContainer : undefined}
      initial="initial"
      animate="animate"
    >
      {children}
    </motion.div>
  );
};

export const AnimatedListItem: React.FC<{
  children: React.ReactNode;
  className?: string;
}> = ({ children, className = '' }) => {
  return (
    <motion.div
      className={className}
      variants={animationPresets.listItem}
    >
      {children}
    </motion.div>
  );
};

// Micro-interaction Components
export const PulseIndicator: React.FC<{
  className?: string;
  color?: string;
}> = ({ className = '', color = 'bg-blue-500' }) => {
  return (
    <motion.div
      className={`w-3 h-3 rounded-full ${color} ${className}`}
      {...animationPresets.pulse}
    />
  );
};

export const SuccessFeedback: React.FC<{
  show: boolean;
  message: string;
  onComplete?: () => void;
}> = ({ show, message, onComplete }) => {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          className="fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50"
          {...animationPresets.successFeedback}
          onAnimationComplete={onComplete}
        >
          {message}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Export all animation utilities
export {
  motion,
  AnimatePresence,
  useSpring,
  useMotionValue,
  useTransform
};

