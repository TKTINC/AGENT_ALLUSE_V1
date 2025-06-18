// Utility functions for data formatting, calculations, and chart helpers

// Currency formatting utilities
export const formatCurrency = (amount: number, options: Intl.NumberFormatOptions = {}): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options
  }).format(amount);
};

export const formatCompactCurrency = (amount: number): string => {
  if (amount >= 1000000) {
    return formatCurrency(amount / 1000000, { maximumFractionDigits: 1 }) + 'M';
  }
  if (amount >= 1000) {
    return formatCurrency(amount / 1000, { maximumFractionDigits: 1 }) + 'K';
  }
  return formatCurrency(amount);
};

// Percentage formatting utilities
export const formatPercentage = (value: number, decimals: number = 2): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

export const formatPercentageChange = (value: number, decimals: number = 2): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${formatPercentage(value, decimals)}`;
};

// Date formatting utilities
export const formatDate = (date: string | Date, format: 'short' | 'medium' | 'long' = 'medium'): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  const options: Intl.DateTimeFormatOptions = {
    short: { month: 'short', day: 'numeric' },
    medium: { month: 'short', day: 'numeric', year: 'numeric' },
    long: { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }
  };

  return dateObj.toLocaleDateString('en-US', options[format]);
};

export const formatTime = (date: string | Date): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleTimeString('en-US', { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: true 
  });
};

// Financial calculations
export const calculateReturn = (initialValue: number, finalValue: number): number => {
  return (finalValue - initialValue) / initialValue;
};

export const calculateAnnualizedReturn = (totalReturn: number, years: number): number => {
  return Math.pow(1 + totalReturn, 1 / years) - 1;
};

export const calculateSharpeRatio = (returns: number[], riskFreeRate: number = 0.02): number => {
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  
  return stdDev === 0 ? 0 : (avgReturn - riskFreeRate) : stdDev;
};

export const calculateMaxDrawdown = (values: number[]): number => {
  let maxDrawdown = 0;
  let peak = values[0];
  
  for (let i = 1; i < values.length; i++) {
    if (values[i] > peak) {
      peak = values[i];
    } else {
      const drawdown = (peak - values[i]) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
  }
  
  return maxDrawdown;
};

export const calculateVolatility = (returns: number[]): number => {
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  return Math.sqrt(variance);
};

export const calculateWinRate = (returns: number[]): number => {
  const wins = returns.filter(r => r > 0).length;
  return wins / returns.length;
};

export const calculateProfitFactor = (returns: number[]): number => {
  const profits = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0);
  const losses = Math.abs(returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0));
  return losses === 0 ? Infinity : profits / losses;
};

// Data transformation utilities
export const aggregateDataByTimeframe = (
  data: any[], 
  timeframe: 'daily' | 'weekly' | 'monthly' | 'quarterly',
  valueKey: string = 'value'
): any[] => {
  const grouped = data.reduce((acc, item) => {
    const date = new Date(item.date);
    let key: string;
    
    switch (timeframe) {
      case 'weekly':
        const weekStart = new Date(date);
        weekStart.setDate(date.getDate() - date.getDay());
        key = weekStart.toISOString().split('T')[0];
        break;
      case 'monthly':
        key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        break;
      case 'quarterly':
        const quarter = Math.floor(date.getMonth() / 3) + 1;
        key = `${date.getFullYear()}-Q${quarter}`;
        break;
      default:
        key = item.date;
    }
    
    if (!acc[key]) {
      acc[key] = { date: key, values: [], count: 0 };
    }
    
    acc[key].values.push(item[valueKey]);
    acc[key].count++;
    
    return acc;
  }, {} as Record<string, any>);
  
  return Object.values(grouped).map((group: any) => ({
    date: group.date,
    [valueKey]: group.values[group.values.length - 1], // Last value for the period
    average: group.values.reduce((sum: number, val: number) => sum + val, 0) / group.count,
    min: Math.min(...group.values),
    max: Math.max(...group.values),
    count: group.count
  }));
};

export const smoothData = (data: number[], windowSize: number = 5): number[] => {
  const smoothed: number[] = [];
  
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(data.length, i + Math.floor(windowSize / 2) + 1);
    const window = data.slice(start, end);
    const average = window.reduce((sum, val) => sum + val, 0) / window.length;
    smoothed.push(average);
  }
  
  return smoothed;
};

// Chart color utilities
export const getColorPalette = (type: 'primary' | 'secondary' | 'risk' | 'performance' = 'primary'): string[] => {
  const palettes = {
    primary: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'],
    secondary: ['#6B7280', '#9CA3AF', '#D1D5DB', '#E5E7EB', '#F3F4F6', '#F9FAFB'],
    risk: ['#10B981', '#F59E0B', '#EF4444'], // Green, Yellow, Red
    performance: ['#10B981', '#3B82F6', '#8B5CF6', '#06B6D4', '#F59E0B', '#EF4444']
  };
  
  return palettes[type];
};

export const getColorByValue = (
  value: number, 
  thresholds: { low: number; high: number },
  colors: { low: string; medium: string; high: string } = {
    low: '#10B981',
    medium: '#F59E0B', 
    high: '#EF4444'
  }
): string => {
  if (value <= thresholds.low) return colors.low;
  if (value <= thresholds.high) return colors.medium;
  return colors.high;
};

// Data validation utilities
export const validateChartData = (data: any[], requiredFields: string[]): boolean => {
  if (!Array.isArray(data) || data.length === 0) return false;
  
  return data.every(item => 
    requiredFields.every(field => 
      item.hasOwnProperty(field) && item[field] !== null && item[field] !== undefined
    )
  );
};

export const sanitizeChartData = (data: any[]): any[] => {
  return data.filter(item => {
    // Remove items with null/undefined critical values
    const hasValidDate = item.date && !isNaN(new Date(item.date).getTime());
    const hasValidNumericValues = Object.values(item).some(value => 
      typeof value === 'number' && !isNaN(value)
    );
    
    return hasValidDate && hasValidNumericValues;
  });
};

// Export utilities
export const exportChartData = (data: any[], filename: string, format: 'csv' | 'json' = 'csv'): void => {
  if (format === 'csv') {
    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(item => Object.values(item).join(','));
    const csvContent = [headers, ...rows].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  } else {
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }
};

// Performance optimization utilities
export const memoizeCalculation = <T extends (...args: any[]) => any>(fn: T): T => {
  const cache = new Map();
  
  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = JSON.stringify(args);
    
    if (cache.has(key)) {
      return cache.get(key);
    }
    
    const result = fn(...args);
    cache.set(key, result);
    
    return result;
  }) as T;
};

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void => {
  let timeoutId: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

// Chart responsiveness utilities
export const getResponsiveChartHeight = (containerWidth: number): number => {
  if (containerWidth < 640) return 250; // Mobile
  if (containerWidth < 1024) return 300; // Tablet
  return 400; // Desktop
};

export const getResponsiveMargins = (containerWidth: number) => {
  if (containerWidth < 640) {
    return { top: 10, right: 10, bottom: 30, left: 40 };
  }
  if (containerWidth < 1024) {
    return { top: 15, right: 20, bottom: 40, left: 50 };
  }
  return { top: 20, right: 30, bottom: 50, left: 60 };
};

// Accessibility utilities
export const generateChartDescription = (
  chartType: string,
  data: any[],
  title: string
): string => {
  const dataPoints = data.length;
  const timeRange = data.length > 0 ? 
    `from ${formatDate(data[0].date)} to ${formatDate(data[data.length - 1].date)}` : 
    'no time range';
  
  return `${chartType} chart titled "${title}" showing ${dataPoints} data points ${timeRange}`;
};

export const getChartAriaLabel = (
  chartType: string,
  title: string,
  summary: string
): string => {
  return `${chartType} chart: ${title}. ${summary}`;
};

// Data generation utilities for testing and demos
export const generateMockPortfolioData = (days: number = 30): any[] => {
  const data = [];
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);
  
  let totalValue = 100000;
  let generationAccount = 40000;
  let revenueAccount = 35000;
  let compoundingAccount = 25000;
  
  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + i);
    
    // Simulate daily changes
    const dailyReturn = (Math.random() - 0.5) * 0.04; // ±2% daily change
    const change = totalValue * dailyReturn;
    
    totalValue += change;
    generationAccount += change * 0.4;
    revenueAccount += change * 0.35;
    compoundingAccount += change * 0.25;
    
    const cumulativeReturn = (totalValue - 100000) / 100000;
    
    data.push({
      date: date.toISOString().split('T')[0],
      totalValue: Math.round(totalValue),
      generationAccount: Math.round(generationAccount),
      revenueAccount: Math.round(revenueAccount),
      compoundingAccount: Math.round(compoundingAccount),
      dailyReturn,
      cumulativeReturn,
      weekClassification: dailyReturn > 0.01 ? 'Green' : dailyReturn < -0.01 ? 'Red' : 'Yellow'
    });
  }
  
  return data;
};

export const generateMockWeekClassificationData = (weeks: number = 12): any[] => {
  const data = [];
  
  for (let i = 1; i <= weeks; i++) {
    const returnValue = (Math.random() - 0.3) * 0.1; // Slightly positive bias
    const classification = returnValue > 0.02 ? 'Green' : returnValue < -0.02 ? 'Red' : 'Yellow';
    
    data.push({
      week: i,
      classification,
      return: returnValue,
      trades: Math.floor(Math.random() * 20) + 5,
      compliance: 0.8 + Math.random() * 0.2 // 80-100% compliance
    });
  }
  
  return data;
};

export const generateMockRiskData = (): any[] => {
  const accounts = ['Generation', 'Revenue', 'Compounding'];
  const timeframes = ['1W', '1M', '3M', '1Y'];
  const data = [];
  
  accounts.forEach(account => {
    timeframes.forEach(timeframe => {
      data.push({
        account,
        timeframe,
        riskLevel: Math.random(),
        volatility: Math.random() * 0.3,
        maxDrawdown: Math.random() * 0.2
      });
    });
  });
  
  return data;
};

export const generateMockTradingOpportunities = (): any[] => {
  const symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VIX'];
  const strategies = ['Covered Call', 'Cash Secured Put', 'Iron Condor', 'Butterfly Spread'];
  const riskLevels = ['Low', 'Medium', 'High'];
  
  return symbols.map(symbol => ({
    symbol,
    probability: 0.3 + Math.random() * 0.6,
    expectedReturn: (Math.random() - 0.2) * 0.15,
    riskLevel: riskLevels[Math.floor(Math.random() * riskLevels.length)],
    timeframe: `${Math.floor(Math.random() * 30) + 1} days`,
    strategy: strategies[Math.floor(Math.random() * strategies.length)]
  }));
};

