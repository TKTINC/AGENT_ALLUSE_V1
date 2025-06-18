import { useState, useEffect, useCallback, useRef } from 'react';

// Enhanced useLocalStorage hook with type safety and error handling
export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue] as const;
}

// Enhanced useDebounce hook for performance optimization
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

// Enhanced useAsync hook for handling async operations
interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

export function useAsync<T>(
  asyncFunction: () => Promise<T>,
  dependencies: React.DependencyList = []
) {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null
  });

  const execute = useCallback(async () => {
    setState({ data: null, loading: true, error: null });
    try {
      const data = await asyncFunction();
      setState({ data, loading: false, error: null });
    } catch (error) {
      setState({ data: null, loading: false, error: error as Error });
    }
  }, dependencies);

  useEffect(() => {
    execute();
  }, [execute]);

  return { ...state, execute };
}

// Enhanced useIntersectionObserver hook for lazy loading and animations
export function useIntersectionObserver(
  elementRef: React.RefObject<Element>,
  options: IntersectionObserverInit = {}
) {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasIntersected, setHasIntersected] = useState(false);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(([entry]) => {
      setIsIntersecting(entry.isIntersecting);
      if (entry.isIntersecting && !hasIntersected) {
        setHasIntersected(true);
      }
    }, options);

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [elementRef, options, hasIntersected]);

  return { isIntersecting, hasIntersected };
}

// Enhanced useWindowSize hook for responsive design
export function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0
  });

  useEffect(() => {
    function handleResize() {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    }

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return windowSize;
}

// Enhanced useClickOutside hook for modals and dropdowns
export function useClickOutside<T extends HTMLElement>(
  ref: React.RefObject<T>,
  handler: () => void
) {
  useEffect(() => {
    const listener = (event: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) {
        return;
      }
      handler();
    };

    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);

    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler]);
}

// Enhanced useKeyPress hook for keyboard shortcuts
export function useKeyPress(targetKey: string, callback: () => void) {
  useEffect(() => {
    const downHandler = (event: KeyboardEvent) => {
      if (event.key === targetKey) {
        callback();
      }
    };

    window.addEventListener('keydown', downHandler);
    return () => {
      window.removeEventListener('keydown', downHandler);
    };
  }, [targetKey, callback]);
}

// Enhanced usePrevious hook for comparing previous values
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}

// Enhanced useToggle hook for boolean state management
export function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);
  
  const toggle = useCallback(() => setValue(v => !v), []);
  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);

  return [value, { toggle, setTrue, setFalse }] as const;
}

// Enhanced useInterval hook for periodic updates
export function useInterval(callback: () => void, delay: number | null) {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    const id = setInterval(() => savedCallback.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

// Enhanced useMediaQuery hook for responsive design
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(query);
    if (media.matches !== matches) {
      setMatches(media.matches);
    }

    const listener = () => setMatches(media.matches);
    media.addEventListener('change', listener);
    
    return () => media.removeEventListener('change', listener);
  }, [matches, query]);

  return matches;
}

// Enhanced useForm hook for form state management
interface FormState<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isValid: boolean;
  isSubmitting: boolean;
}

interface FormConfig<T> {
  initialValues: T;
  validate?: (values: T) => Partial<Record<keyof T, string>>;
  onSubmit: (values: T) => Promise<void> | void;
}

export function useForm<T extends Record<string, any>>({
  initialValues,
  validate,
  onSubmit
}: FormConfig<T>) {
  const [state, setState] = useState<FormState<T>>({
    values: initialValues,
    errors: {},
    touched: {},
    isValid: true,
    isSubmitting: false
  });

  const setValue = useCallback((name: keyof T, value: any) => {
    setState(prev => ({
      ...prev,
      values: { ...prev.values, [name]: value },
      touched: { ...prev.touched, [name]: true }
    }));
  }, []);

  const setError = useCallback((name: keyof T, error: string) => {
    setState(prev => ({
      ...prev,
      errors: { ...prev.errors, [name]: error }
    }));
  }, []);

  const clearError = useCallback((name: keyof T) => {
    setState(prev => ({
      ...prev,
      errors: { ...prev.errors, [name]: undefined }
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      values: initialValues,
      errors: {},
      touched: {},
      isValid: true,
      isSubmitting: false
    });
  }, [initialValues]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    setState(prev => ({ ...prev, isSubmitting: true }));
    
    try {
      if (validate) {
        const errors = validate(state.values);
        const hasErrors = Object.values(errors).some(error => error);
        
        setState(prev => ({
          ...prev,
          errors,
          isValid: !hasErrors
        }));
        
        if (hasErrors) {
          setState(prev => ({ ...prev, isSubmitting: false }));
          return;
        }
      }
      
      await onSubmit(state.values);
    } catch (error) {
      console.error('Form submission error:', error);
    } finally {
      setState(prev => ({ ...prev, isSubmitting: false }));
    }
  }, [state.values, validate, onSubmit]);

  return {
    ...state,
    setValue,
    setError,
    clearError,
    reset,
    handleSubmit
  };
}

// Enhanced useNotifications hook for toast management
interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((notification: Omit<Notification, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    setNotifications(prev => [...prev, { ...notification, id }]);
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return {
    notifications,
    addNotification,
    removeNotification,
    clearNotifications
  };
}

// Enhanced useTheme hook for theme management
interface Theme {
  mode: 'light' | 'dark';
  colors: Record<string, string>;
  fonts: Record<string, string>;
  spacing: Record<string, string>;
}

export function useTheme() {
  const [theme, setTheme] = useLocalStorage<Theme>('theme', {
    mode: 'light',
    colors: {
      primary: '#3B82F6',
      secondary: '#6B7280',
      success: '#10B981',
      warning: '#F59E0B',
      error: '#EF4444'
    },
    fonts: {
      sans: 'Inter, system-ui, sans-serif',
      mono: 'Monaco, Consolas, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    }
  });

  const toggleMode = useCallback(() => {
    setTheme(prev => ({
      ...prev,
      mode: prev.mode === 'light' ? 'dark' : 'light'
    }));
  }, [setTheme]);

  const updateColors = useCallback((colors: Partial<Theme['colors']>) => {
    setTheme(prev => ({
      ...prev,
      colors: { ...prev.colors, ...colors }
    }));
  }, [setTheme]);

  return {
    theme,
    toggleMode,
    updateColors,
    setTheme
  };
}

