// User Experience Enhancement Library
// Advanced UX features including animations, accessibility, and interactions

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSpring, animated, useTransition, config } from '@react-spring/web';

// Animation presets for consistent UX
export const animationPresets = {
  fadeIn: {
    from: { opacity: 0 },
    to: { opacity: 1 },
    config: config.gentle
  },
  slideUp: {
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: 1, transform: 'translateY(0px)' },
    config: config.gentle
  },
  slideDown: {
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: 1, transform: 'translateY(0px)' },
    config: config.gentle
  },
  scaleIn: {
    from: { opacity: 0, transform: 'scale(0.9)' },
    to: { opacity: 1, transform: 'scale(1)' },
    config: config.gentle
  },
  bounce: {
    from: { transform: 'scale(1)' },
    to: { transform: 'scale(1.05)' },
    config: config.wobbly
  }
};

// Animated wrapper component
interface AnimatedWrapperProps {
  children: React.ReactNode;
  animation: keyof typeof animationPresets;
  delay?: number;
  trigger?: boolean;
  className?: string;
}

export const AnimatedWrapper: React.FC<AnimatedWrapperProps> = ({
  children,
  animation,
  delay = 0,
  trigger = true,
  className
}) => {
  const styles = useSpring({
    ...animationPresets[animation],
    delay,
    reset: !trigger,
    reverse: !trigger
  });

  return (
    <animated.div style={styles} className={className}>
      {children}
    </animated.div>
  );
};

// Staggered animation for lists
interface StaggeredListProps {
  children: React.ReactNode[];
  staggerDelay?: number;
  animation?: keyof typeof animationPresets;
}

export const StaggeredList: React.FC<StaggeredListProps> = ({
  children,
  staggerDelay = 100,
  animation = 'slideUp'
}) => {
  return (
    <>
      {children.map((child, index) => (
        <AnimatedWrapper
          key={index}
          animation={animation}
          delay={index * staggerDelay}
        >
          {child}
        </AnimatedWrapper>
      ))}
    </>
  );
};

// Page transition component
interface PageTransitionProps {
  children: React.ReactNode;
  location: string;
}

export const PageTransition: React.FC<PageTransitionProps> = ({
  children,
  location
}) => {
  const transitions = useTransition(location, {
    from: { opacity: 0, transform: 'translateX(100px)' },
    enter: { opacity: 1, transform: 'translateX(0px)' },
    leave: { opacity: 0, transform: 'translateX(-100px)' },
    config: config.gentle
  });

  return transitions((style, item) => (
    <animated.div style={style} className="absolute inset-0">
      {children}
    </animated.div>
  ));
};

// Loading skeleton component
interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  className?: string;
  animated?: boolean;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = '1rem',
  className = '',
  animated = true
}) => {
  const pulseAnimation = useSpring({
    from: { opacity: 0.4 },
    to: { opacity: 1 },
    config: { duration: 1000 },
    loop: { reverse: true }
  });

  const style = {
    width,
    height,
    backgroundColor: '#e2e8f0',
    borderRadius: '0.375rem'
  };

  if (animated) {
    return (
      <animated.div
        style={{ ...style, ...pulseAnimation }}
        className={className}
      />
    );
  }

  return <div style={style} className={className} />;
};

// Accessibility utilities
export const useAccessibility = () => {
  const [isHighContrast, setIsHighContrast] = useState(false);
  const [isReducedMotion, setIsReducedMotion] = useState(false);
  const [fontSize, setFontSize] = useState(16);

  useEffect(() => {
    // Check for user preferences
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)');
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    setIsHighContrast(highContrastQuery.matches);
    setIsReducedMotion(reducedMotionQuery.matches);

    const handleHighContrastChange = (e: MediaQueryListEvent) => {
      setIsHighContrast(e.matches);
    };

    const handleReducedMotionChange = (e: MediaQueryListEvent) => {
      setIsReducedMotion(e.matches);
    };

    highContrastQuery.addEventListener('change', handleHighContrastChange);
    reducedMotionQuery.addEventListener('change', handleReducedMotionChange);

    return () => {
      highContrastQuery.removeEventListener('change', handleHighContrastChange);
      reducedMotionQuery.removeEventListener('change', handleReducedMotionChange);
    };
  }, []);

  const increaseFontSize = useCallback(() => {
    setFontSize(prev => Math.min(prev + 2, 24));
  }, []);

  const decreaseFontSize = useCallback(() => {
    setFontSize(prev => Math.max(prev - 2, 12));
  }, []);

  const resetFontSize = useCallback(() => {
    setFontSize(16);
  }, []);

  return {
    isHighContrast,
    isReducedMotion,
    fontSize,
    increaseFontSize,
    decreaseFontSize,
    resetFontSize
  };
};

// Focus management
export const useFocusManagement = () => {
  const focusableElements = useRef<HTMLElement[]>([]);
  const currentFocusIndex = useRef(0);

  const registerFocusableElement = useCallback((element: HTMLElement) => {
    if (element && !focusableElements.current.includes(element)) {
      focusableElements.current.push(element);
    }
  }, []);

  const unregisterFocusableElement = useCallback((element: HTMLElement) => {
    const index = focusableElements.current.indexOf(element);
    if (index > -1) {
      focusableElements.current.splice(index, 1);
    }
  }, []);

  const focusNext = useCallback(() => {
    if (focusableElements.current.length === 0) return;
    
    currentFocusIndex.current = (currentFocusIndex.current + 1) % focusableElements.current.length;
    focusableElements.current[currentFocusIndex.current]?.focus();
  }, []);

  const focusPrevious = useCallback(() => {
    if (focusableElements.current.length === 0) return;
    
    currentFocusIndex.current = currentFocusIndex.current === 0 
      ? focusableElements.current.length - 1 
      : currentFocusIndex.current - 1;
    focusableElements.current[currentFocusIndex.current]?.focus();
  }, []);

  const focusFirst = useCallback(() => {
    if (focusableElements.current.length > 0) {
      currentFocusIndex.current = 0;
      focusableElements.current[0]?.focus();
    }
  }, []);

  const focusLast = useCallback(() => {
    if (focusableElements.current.length > 0) {
      currentFocusIndex.current = focusableElements.current.length - 1;
      focusableElements.current[currentFocusIndex.current]?.focus();
    }
  }, []);

  return {
    registerFocusableElement,
    unregisterFocusableElement,
    focusNext,
    focusPrevious,
    focusFirst,
    focusLast
  };
};

// Keyboard navigation hook
export const useKeyboardNavigation = (
  onEnter?: () => void,
  onEscape?: () => void,
  onArrowUp?: () => void,
  onArrowDown?: () => void,
  onArrowLeft?: () => void,
  onArrowRight?: () => void
) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      switch (event.key) {
        case 'Enter':
          onEnter?.();
          break;
        case 'Escape':
          onEscape?.();
          break;
        case 'ArrowUp':
          event.preventDefault();
          onArrowUp?.();
          break;
        case 'ArrowDown':
          event.preventDefault();
          onArrowDown?.();
          break;
        case 'ArrowLeft':
          onArrowLeft?.();
          break;
        case 'ArrowRight':
          onArrowRight?.();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onEnter, onEscape, onArrowUp, onArrowDown, onArrowLeft, onArrowRight]);
};

// Screen reader announcements
export const useScreenReader = () => {
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', priority);
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }, []);

  return { announce };
};

// Touch gesture support
export const useTouchGestures = (
  onSwipeLeft?: () => void,
  onSwipeRight?: () => void,
  onSwipeUp?: () => void,
  onSwipeDown?: () => void,
  threshold: number = 50
) => {
  const touchStart = useRef<{ x: number; y: number } | null>(null);

  const handleTouchStart = useCallback((event: React.TouchEvent) => {
    const touch = event.touches[0];
    touchStart.current = { x: touch.clientX, y: touch.clientY };
  }, []);

  const handleTouchEnd = useCallback((event: React.TouchEvent) => {
    if (!touchStart.current) return;

    const touch = event.changedTouches[0];
    const deltaX = touch.clientX - touchStart.current.x;
    const deltaY = touch.clientY - touchStart.current.y;

    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      // Horizontal swipe
      if (Math.abs(deltaX) > threshold) {
        if (deltaX > 0) {
          onSwipeRight?.();
        } else {
          onSwipeLeft?.();
        }
      }
    } else {
      // Vertical swipe
      if (Math.abs(deltaY) > threshold) {
        if (deltaY > 0) {
          onSwipeDown?.();
        } else {
          onSwipeUp?.();
        }
      }
    }

    touchStart.current = null;
  }, [onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown, threshold]);

  return { handleTouchStart, handleTouchEnd };
};

// Smooth scrolling utility
export const useSmoothScroll = () => {
  const scrollToElement = useCallback((
    elementId: string,
    offset: number = 0,
    behavior: ScrollBehavior = 'smooth'
  ) => {
    const element = document.getElementById(elementId);
    if (element) {
      const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
      window.scrollTo({
        top: elementPosition - offset,
        behavior
      });
    }
  }, []);

  const scrollToTop = useCallback((behavior: ScrollBehavior = 'smooth') => {
    window.scrollTo({ top: 0, behavior });
  }, []);

  return { scrollToElement, scrollToTop };
};

// Theme management
export const useTheme = () => {
  const [theme, setTheme] = useState<'light' | 'dark' | 'auto'>('auto');
  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemTheme(mediaQuery.matches ? 'dark' : 'light');

    const handleChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const effectiveTheme = theme === 'auto' ? systemTheme : theme;

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  return { theme, effectiveTheme, systemTheme, setTheme, toggleTheme };
};

// Error boundary component
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  { hasError: boolean; error: Error | null }
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.props.onError?.(error, errorInfo);
  }

  resetError = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback;
        return <FallbackComponent error={this.state.error} resetError={this.resetError} />;
      }

      return (
        <div className="p-6 text-center">
          <h2 className="text-xl font-semibold text-red-600 mb-2">Something went wrong</h2>
          <p className="text-gray-600 mb-4">An unexpected error occurred. Please try again.</p>
          <button
            onClick={this.resetError}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Notification system
interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const useNotifications = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((notification: Omit<Notification, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newNotification = { ...notification, id };
    
    setNotifications(prev => [...prev, newNotification]);

    if (notification.duration !== 0) {
      setTimeout(() => {
        removeNotification(id);
      }, notification.duration || 5000);
    }

    return id;
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const clearAllNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return {
    notifications,
    addNotification,
    removeNotification,
    clearAllNotifications
  };
};

// Progressive enhancement utilities
export const useProgressiveEnhancement = () => {
  const [isJavaScriptEnabled] = useState(true);
  const [isCSSSupported, setIsCSSSupported] = useState(true);
  const [isWebGLSupported, setIsWebGLSupported] = useState(false);

  useEffect(() => {
    // Check CSS support
    const testElement = document.createElement('div');
    testElement.style.display = 'grid';
    setIsCSSSupported(testElement.style.display === 'grid');

    // Check WebGL support
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    setIsWebGLSupported(!!gl);
  }, []);

  return {
    isJavaScriptEnabled,
    isCSSSupported,
    isWebGLSupported
  };
};

