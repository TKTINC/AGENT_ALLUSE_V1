import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronUp, Check, X, AlertCircle, Info, CheckCircle, XCircle, Loader2 } from 'lucide-react';

// Enhanced Button Component with multiple variants and states
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive' | 'success';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  className = '',
  disabled,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 shadow-sm hover:shadow-md',
    secondary: 'bg-gray-600 text-white hover:bg-gray-700 focus:ring-gray-500 shadow-sm hover:shadow-md',
    outline: 'border-2 border-blue-600 text-blue-600 hover:bg-blue-50 focus:ring-blue-500',
    ghost: 'text-gray-600 hover:bg-gray-100 focus:ring-gray-500',
    destructive: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 shadow-sm hover:shadow-md',
    success: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500 shadow-sm hover:shadow-md'
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-3 text-base gap-2.5',
    xl: 'px-8 py-4 text-lg gap-3'
  };

  const widthClass = fullWidth ? 'w-full' : '';

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${widthClass} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" />}
      {!loading && icon && iconPosition === 'left' && icon}
      {children}
      {!loading && icon && iconPosition === 'right' && icon}
    </button>
  );
};

// Enhanced Card Component with multiple variants and animations
interface CardProps {
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'outlined' | 'filled';
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  hoverable?: boolean;
  clickable?: boolean;
  onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'default',
  padding = 'md',
  className = '',
  hoverable = false,
  clickable = false,
  onClick
}) => {
  const baseClasses = 'rounded-lg transition-all duration-200';
  
  const variantClasses = {
    default: 'bg-white border border-gray-200',
    elevated: 'bg-white shadow-lg border border-gray-100',
    outlined: 'bg-transparent border-2 border-gray-300',
    filled: 'bg-gray-50 border border-gray-200'
  };

  const paddingClasses = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
    xl: 'p-8'
  };

  const interactiveClasses = hoverable || clickable ? 'hover:shadow-md hover:-translate-y-0.5' : '';
  const cursorClass = clickable ? 'cursor-pointer' : '';

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${paddingClasses[padding]} ${interactiveClasses} ${cursorClass} ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
};

// Enhanced Modal Component with animations and accessibility
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: React.ReactNode;
  showCloseButton?: boolean;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  size = 'md',
  children,
  showCloseButton = true,
  closeOnOverlayClick = true,
  closeOnEscape = true
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (closeOnEscape && e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, closeOnEscape, onClose]);

  const sizeClasses = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-full mx-4'
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity duration-300"
        onClick={closeOnOverlayClick ? onClose : undefined}
      />
      
      {/* Modal */}
      <div
        ref={modalRef}
        className={`relative bg-white rounded-lg shadow-xl transform transition-all duration-300 ${sizeClasses[size]} w-full mx-4`}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
            {showCloseButton && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            )}
          </div>
        )}
        
        {/* Content */}
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>
  );
};

// Enhanced Dropdown Component with search and multi-select
interface DropdownOption {
  value: string;
  label: string;
  disabled?: boolean;
  icon?: React.ReactNode;
}

interface DropdownProps {
  options: DropdownOption[];
  value?: string | string[];
  onChange: (value: string | string[]) => void;
  placeholder?: string;
  searchable?: boolean;
  multiSelect?: boolean;
  disabled?: boolean;
  className?: string;
}

export const Dropdown: React.FC<DropdownProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Select an option',
  searchable = false,
  multiSelect = false,
  disabled = false,
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const filteredOptions = searchable
    ? options.filter(option => 
        option.label.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : options;

  const selectedOptions = Array.isArray(value) ? value : value ? [value] : [];
  const selectedLabels = options
    .filter(option => selectedOptions.includes(option.value))
    .map(option => option.label);

  const handleOptionClick = (optionValue: string) => {
    if (multiSelect) {
      const newValue = selectedOptions.includes(optionValue)
        ? selectedOptions.filter(v => v !== optionValue)
        : [...selectedOptions, optionValue];
      onChange(newValue);
    } else {
      onChange(optionValue);
      setIsOpen(false);
    }
  };

  return (
    <div ref={dropdownRef} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`w-full px-4 py-2 text-left bg-white border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-gray-400'
        }`}
      >
        <div className="flex items-center justify-between">
          <span className={selectedLabels.length > 0 ? 'text-gray-900' : 'text-gray-500'}>
            {selectedLabels.length > 0 
              ? multiSelect && selectedLabels.length > 1
                ? `${selectedLabels.length} selected`
                : selectedLabels.join(', ')
              : placeholder
            }
          </span>
          {isOpen ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </div>
      </button>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg">
          {searchable && (
            <div className="p-2 border-b border-gray-200">
              <input
                type="text"
                placeholder="Search options..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
          
          <div className="max-h-60 overflow-y-auto">
            {filteredOptions.length === 0 ? (
              <div className="px-4 py-2 text-gray-500">No options found</div>
            ) : (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => !option.disabled && handleOptionClick(option.value)}
                  disabled={option.disabled}
                  className={`w-full px-4 py-2 text-left hover:bg-gray-50 focus:outline-none focus:bg-gray-50 transition-colors flex items-center gap-2 ${
                    option.disabled ? 'opacity-50 cursor-not-allowed' : ''
                  } ${
                    selectedOptions.includes(option.value) ? 'bg-blue-50 text-blue-700' : 'text-gray-900'
                  }`}
                >
                  {multiSelect && (
                    <div className={`w-4 h-4 border rounded flex items-center justify-center ${
                      selectedOptions.includes(option.value) 
                        ? 'bg-blue-600 border-blue-600' 
                        : 'border-gray-300'
                    }`}>
                      {selectedOptions.includes(option.value) && (
                        <Check className="w-3 h-3 text-white" />
                      )}
                    </div>
                  )}
                  {option.icon && option.icon}
                  <span>{option.label}</span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Enhanced Toast Notification System
interface ToastProps {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  onClose: (id: string) => void;
}

export const Toast: React.FC<ToastProps> = ({
  id,
  type,
  title,
  message,
  duration = 5000,
  onClose
}) => {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => onClose(id), duration);
      return () => clearTimeout(timer);
    }
  }, [id, duration, onClose]);

  const typeConfig = {
    success: {
      icon: <CheckCircle className="w-5 h-5" />,
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      iconColor: 'text-green-600',
      titleColor: 'text-green-800',
      messageColor: 'text-green-700'
    },
    error: {
      icon: <XCircle className="w-5 h-5" />,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      iconColor: 'text-red-600',
      titleColor: 'text-red-800',
      messageColor: 'text-red-700'
    },
    warning: {
      icon: <AlertCircle className="w-5 h-5" />,
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      iconColor: 'text-yellow-600',
      titleColor: 'text-yellow-800',
      messageColor: 'text-yellow-700'
    },
    info: {
      icon: <Info className="w-5 h-5" />,
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      iconColor: 'text-blue-600',
      titleColor: 'text-blue-800',
      messageColor: 'text-blue-700'
    }
  };

  const config = typeConfig[type];

  return (
    <div className={`max-w-sm w-full ${config.bgColor} border ${config.borderColor} rounded-lg shadow-lg p-4 transform transition-all duration-300 ease-in-out`}>
      <div className="flex items-start">
        <div className={`flex-shrink-0 ${config.iconColor}`}>
          {config.icon}
        </div>
        <div className="ml-3 w-0 flex-1">
          <p className={`text-sm font-medium ${config.titleColor}`}>
            {title}
          </p>
          {message && (
            <p className={`mt-1 text-sm ${config.messageColor}`}>
              {message}
            </p>
          )}
        </div>
        <div className="ml-4 flex-shrink-0 flex">
          <button
            onClick={() => onClose(id)}
            className={`rounded-md inline-flex text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Enhanced Loading Spinner Component
interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'blue' | 'gray' | 'green' | 'red' | 'yellow';
  text?: string;
  overlay?: boolean;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  color = 'blue',
  text,
  overlay = false
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const colorClasses = {
    blue: 'text-blue-600',
    gray: 'text-gray-600',
    green: 'text-green-600',
    red: 'text-red-600',
    yellow: 'text-yellow-600'
  };

  const spinner = (
    <div className="flex flex-col items-center gap-3">
      <Loader2 className={`${sizeClasses[size]} ${colorClasses[color]} animate-spin`} />
      {text && (
        <p className={`text-sm ${colorClasses[color]} font-medium`}>
          {text}
        </p>
      )}
    </div>
  );

  if (overlay) {
    return (
      <div className="fixed inset-0 bg-white bg-opacity-75 flex items-center justify-center z-50">
        {spinner}
      </div>
    );
  }

  return spinner;
};

// Enhanced Progress Bar Component
interface ProgressBarProps {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'green' | 'yellow' | 'red';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  size = 'md',
  color = 'blue',
  showLabel = false,
  label,
  animated = false
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const sizeClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4'
  };

  const colorClasses = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600'
  };

  return (
    <div className="w-full">
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">
            {label || `${Math.round(percentage)}%`}
          </span>
          {showLabel && !label && (
            <span className="text-sm text-gray-500">
              {value} / {max}
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-gray-200 rounded-full ${sizeClasses[size]}`}>
        <div
          className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full transition-all duration-300 ease-out ${
            animated ? 'animate-pulse' : ''
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

// Enhanced Tabs Component
interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  badge?: string | number;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (tabId: string) => void;
  variant?: 'default' | 'pills' | 'underline';
  size?: 'sm' | 'md' | 'lg';
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onChange,
  variant = 'default',
  size = 'md'
}) => {
  const baseClasses = 'flex items-center gap-2 font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2';
  
  const variantClasses = {
    default: {
      container: 'border-b border-gray-200',
      tab: 'px-4 py-2 border-b-2 border-transparent hover:border-gray-300',
      active: 'border-blue-600 text-blue-600',
      inactive: 'text-gray-600 hover:text-gray-900'
    },
    pills: {
      container: 'bg-gray-100 rounded-lg p-1',
      tab: 'px-3 py-1.5 rounded-md',
      active: 'bg-white text-blue-600 shadow-sm',
      inactive: 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'
    },
    underline: {
      container: '',
      tab: 'px-4 py-2 border-b-2 border-transparent',
      active: 'border-blue-600 text-blue-600',
      inactive: 'text-gray-600 hover:text-gray-900 hover:border-gray-300'
    }
  };

  const sizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  };

  const config = variantClasses[variant];

  return (
    <div className={config.container}>
      <nav className="flex space-x-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => !tab.disabled && onChange(tab.id)}
            disabled={tab.disabled}
            className={`${baseClasses} ${config.tab} ${sizeClasses[size]} ${
              activeTab === tab.id ? config.active : config.inactive
            } ${tab.disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {tab.icon && tab.icon}
            <span>{tab.label}</span>
            {tab.badge && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-gray-200 text-gray-700 rounded-full">
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </nav>
    </div>
  );
};

