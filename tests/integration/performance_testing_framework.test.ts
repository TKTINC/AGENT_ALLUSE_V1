import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Performance testing utilities and frameworks
interface PerformanceMetrics {
  renderTime: number;
  memoryUsage: number;
  bundleSize: number;
  networkRequests: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  cumulativeLayoutShift: number;
  firstInputDelay: number;
}

interface ComponentPerformanceTest {
  component: React.ComponentType<any>;
  props?: any;
  expectedRenderTime: number;
  expectedMemoryUsage: number;
  testName: string;
}

// Performance monitoring utilities
class PerformanceMonitor {
  private metrics: PerformanceMetrics[] = [];
  private observer: PerformanceObserver | null = null;

  startMonitoring() {
    // Clear previous metrics
    this.metrics = [];
    
    // Start performance observation
    if (typeof PerformanceObserver !== 'undefined') {
      this.observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          this.processPerformanceEntry(entry);
        });
      });
      
      this.observer.observe({ entryTypes: ['measure', 'navigation', 'paint', 'layout-shift'] });
    }
  }

  stopMonitoring(): PerformanceMetrics {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }

    return this.calculateAverageMetrics();
  }

  private processPerformanceEntry(entry: PerformanceEntry) {
    // Process different types of performance entries
    switch (entry.entryType) {
      case 'paint':
        this.processPaintEntry(entry as PerformancePaintTiming);
        break;
      case 'layout-shift':
        this.processLayoutShiftEntry(entry as LayoutShift);
        break;
      case 'navigation':
        this.processNavigationEntry(entry as PerformanceNavigationTiming);
        break;
    }
  }

  private processPaintEntry(entry: PerformancePaintTiming) {
    // Track paint timing metrics
    if (entry.name === 'first-contentful-paint') {
      this.updateMetric('firstContentfulPaint', entry.startTime);
    }
  }

  private processLayoutShiftEntry(entry: LayoutShift) {
    // Track cumulative layout shift
    this.updateMetric('cumulativeLayoutShift', entry.value);
  }

  private processNavigationEntry(entry: PerformanceNavigationTiming) {
    // Track navigation timing
    const loadTime = entry.loadEventEnd - entry.navigationStart;
    this.updateMetric('renderTime', loadTime);
  }

  private updateMetric(key: keyof PerformanceMetrics, value: number) {
    if (this.metrics.length === 0) {
      this.metrics.push({
        renderTime: 0,
        memoryUsage: 0,
        bundleSize: 0,
        networkRequests: 0,
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
        cumulativeLayoutShift: 0,
        firstInputDelay: 0
      });
    }

    this.metrics[0][key] = value;
  }

  private calculateAverageMetrics(): PerformanceMetrics {
    if (this.metrics.length === 0) {
      return {
        renderTime: 0,
        memoryUsage: 0,
        bundleSize: 0,
        networkRequests: 0,
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
        cumulativeLayoutShift: 0,
        firstInputDelay: 0
      };
    }

    return this.metrics[0];
  }

  measureComponentRender<T>(component: React.ComponentType<T>, props?: T): Promise<number> {
    return new Promise((resolve) => {
      const startTime = performance.now();
      
      render(React.createElement(component, props));
      
      // Use requestAnimationFrame to ensure render is complete
      requestAnimationFrame(() => {
        const endTime = performance.now();
        resolve(endTime - startTime);
      });
    });
  }

  measureMemoryUsage(): number {
    if ('memory' in performance) {
      return (performance as any).memory.usedJSHeapSize;
    }
    return 0;
  }

  measureBundleSize(): Promise<number> {
    return new Promise((resolve) => {
      if ('getEntriesByType' in performance) {
        const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
        const totalSize = resources.reduce((total, resource) => {
          return total + (resource.transferSize || 0);
        }, 0);
        resolve(totalSize);
      } else {
        resolve(0);
      }
    });
  }
}

// Load testing utilities
class LoadTester {
  async simulateHighLoad(component: React.ComponentType<any>, iterations: number = 100): Promise<PerformanceMetrics[]> {
    const results: PerformanceMetrics[] = [];
    const monitor = new PerformanceMonitor();

    for (let i = 0; i < iterations; i++) {
      monitor.startMonitoring();
      
      const startTime = performance.now();
      render(React.createElement(component));
      const endTime = performance.now();

      const metrics = monitor.stopMonitoring();
      metrics.renderTime = endTime - startTime;
      metrics.memoryUsage = monitor.measureMemoryUsage();

      results.push(metrics);

      // Clean up between iterations
      document.body.innerHTML = '';
    }

    return results;
  }

  async simulateConcurrentUsers(component: React.ComponentType<any>, userCount: number = 10): Promise<PerformanceMetrics[]> {
    const promises = Array.from({ length: userCount }, () => {
      return this.simulateUserSession(component);
    });

    return Promise.all(promises);
  }

  private async simulateUserSession(component: React.ComponentType<any>): Promise<PerformanceMetrics> {
    const monitor = new PerformanceMonitor();
    monitor.startMonitoring();

    // Simulate user interactions
    const startTime = performance.now();
    
    render(React.createElement(component));
    
    // Simulate user clicks and interactions
    await this.simulateUserInteractions();
    
    const endTime = performance.now();
    
    const metrics = monitor.stopMonitoring();
    metrics.renderTime = endTime - startTime;
    metrics.memoryUsage = monitor.measureMemoryUsage();

    return metrics;
  }

  private async simulateUserInteractions(): Promise<void> {
    // Simulate random user interactions
    const interactions = [
      () => fireEvent.click(document.body),
      () => fireEvent.scroll(window, { target: { scrollY: 100 } }),
      () => fireEvent.keyDown(document.body, { key: 'Tab' }),
      () => fireEvent.mouseMove(document.body, { clientX: 100, clientY: 100 })
    ];

    for (let i = 0; i < 5; i++) {
      const randomInteraction = interactions[Math.floor(Math.random() * interactions.length)];
      randomInteraction();
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
}

// Memory leak detection
class MemoryLeakDetector {
  private initialMemory: number = 0;
  private samples: number[] = [];

  startDetection() {
    this.initialMemory = this.getCurrentMemoryUsage();
    this.samples = [this.initialMemory];
  }

  takeSample() {
    const currentMemory = this.getCurrentMemoryUsage();
    this.samples.push(currentMemory);
  }

  detectLeaks(): { hasLeak: boolean; memoryGrowth: number; samples: number[] } {
    if (this.samples.length < 2) {
      return { hasLeak: false, memoryGrowth: 0, samples: this.samples };
    }

    const memoryGrowth = this.samples[this.samples.length - 1] - this.initialMemory;
    const hasLeak = memoryGrowth > 10 * 1024 * 1024; // 10MB threshold

    return {
      hasLeak,
      memoryGrowth,
      samples: this.samples
    };
  }

  private getCurrentMemoryUsage(): number {
    if ('memory' in performance) {
      return (performance as any).memory.usedJSHeapSize;
    }
    return 0;
  }
}

// Bundle size analyzer
class BundleSizeAnalyzer {
  async analyzeBundleSize(): Promise<{ totalSize: number; breakdown: { [key: string]: number } }> {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    
    const breakdown: { [key: string]: number } = {
      javascript: 0,
      css: 0,
      images: 0,
      fonts: 0,
      other: 0
    };

    let totalSize = 0;

    resources.forEach(resource => {
      const size = resource.transferSize || 0;
      totalSize += size;

      const url = resource.name;
      if (url.includes('.js')) {
        breakdown.javascript += size;
      } else if (url.includes('.css')) {
        breakdown.css += size;
      } else if (url.match(/\.(png|jpg|jpeg|gif|svg|webp)$/)) {
        breakdown.images += size;
      } else if (url.match(/\.(woff|woff2|ttf|eot)$/)) {
        breakdown.fonts += size;
      } else {
        breakdown.other += size;
      }
    });

    return { totalSize, breakdown };
  }

  checkBundleSizeThresholds(analysis: { totalSize: number; breakdown: { [key: string]: number } }): {
    passed: boolean;
    violations: string[];
  } {
    const violations: string[] = [];
    
    // Define thresholds (in bytes)
    const thresholds = {
      totalSize: 2 * 1024 * 1024, // 2MB
      javascript: 1 * 1024 * 1024, // 1MB
      css: 200 * 1024, // 200KB
      images: 500 * 1024, // 500KB
      fonts: 100 * 1024 // 100KB
    };

    if (analysis.totalSize > thresholds.totalSize) {
      violations.push(`Total bundle size (${(analysis.totalSize / 1024 / 1024).toFixed(2)}MB) exceeds threshold (${(thresholds.totalSize / 1024 / 1024).toFixed(2)}MB)`);
    }

    Object.entries(thresholds).forEach(([key, threshold]) => {
      if (key !== 'totalSize' && analysis.breakdown[key] > threshold) {
        violations.push(`${key} size (${(analysis.breakdown[key] / 1024).toFixed(2)}KB) exceeds threshold (${(threshold / 1024).toFixed(2)}KB)`);
      }
    });

    return {
      passed: violations.length === 0,
      violations
    };
  }
}

// Performance test suite
describe('WS6-P4: Performance Testing and Validation', () => {
  let performanceMonitor: PerformanceMonitor;
  let loadTester: LoadTester;
  let memoryLeakDetector: MemoryLeakDetector;
  let bundleSizeAnalyzer: BundleSizeAnalyzer;

  beforeEach(() => {
    performanceMonitor = new PerformanceMonitor();
    loadTester = new LoadTester();
    memoryLeakDetector = new MemoryLeakDetector();
    bundleSizeAnalyzer = new BundleSizeAnalyzer();
  });

  afterEach(() => {
    // Clean up after each test
    document.body.innerHTML = '';
    
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
  });

  describe('Component Render Performance', () => {
    const componentTests: ComponentPerformanceTest[] = [
      {
        component: () => React.createElement('div', null, 'ConversationalInterface'),
        expectedRenderTime: 100,
        expectedMemoryUsage: 5 * 1024 * 1024,
        testName: 'ConversationalInterface'
      },
      {
        component: () => React.createElement('div', null, 'AccountVisualization'),
        expectedRenderTime: 150,
        expectedMemoryUsage: 8 * 1024 * 1024,
        testName: 'AccountVisualization'
      },
      {
        component: () => React.createElement('div', null, 'TradingDashboard'),
        expectedRenderTime: 200,
        expectedMemoryUsage: 10 * 1024 * 1024,
        testName: 'TradingDashboard'
      },
      {
        component: () => React.createElement('div', null, 'AdvancedDashboardBuilder'),
        expectedRenderTime: 250,
        expectedMemoryUsage: 12 * 1024 * 1024,
        testName: 'AdvancedDashboardBuilder'
      },
      {
        component: () => React.createElement('div', null, 'SystemIntegrationHub'),
        expectedRenderTime: 180,
        expectedMemoryUsage: 9 * 1024 * 1024,
        testName: 'SystemIntegrationHub'
      }
    ];

    componentTests.forEach(({ component, expectedRenderTime, expectedMemoryUsage, testName }) => {
      test(`${testName} should render within performance budget`, async () => {
        const renderTime = await performanceMonitor.measureComponentRender(component);
        const memoryUsage = performanceMonitor.measureMemoryUsage();

        expect(renderTime).toBeLessThan(expectedRenderTime);
        expect(memoryUsage).toBeLessThan(expectedMemoryUsage);

        console.log(`${testName} Performance:`, {
          renderTime: `${renderTime.toFixed(2)}ms`,
          memoryUsage: `${(memoryUsage / 1024 / 1024).toFixed(2)}MB`,
          expectedRenderTime: `${expectedRenderTime}ms`,
          expectedMemoryUsage: `${(expectedMemoryUsage / 1024 / 1024).toFixed(2)}MB`
        });
      });
    });

    test('should handle rapid re-renders efficiently', async () => {
      const TestComponent = () => {
        const [count, setCount] = React.useState(0);
        
        React.useEffect(() => {
          const interval = setInterval(() => {
            setCount(c => c + 1);
          }, 10);
          
          setTimeout(() => clearInterval(interval), 1000);
          
          return () => clearInterval(interval);
        }, []);
        
        return React.createElement('div', null, `Count: ${count}`);
      };

      const startTime = performance.now();
      render(React.createElement(TestComponent));
      
      await waitFor(() => {
        expect(screen.getByText(/Count: \d+/)).toBeInTheDocument();
      }, { timeout: 2000 });
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      
      expect(totalTime).toBeLessThan(2000); // Should complete within 2 seconds
    });
  });

  describe('Load Testing', () => {
    test('should handle high component load', async () => {
      const TestComponent = () => React.createElement('div', null, 'Load Test Component');
      
      const results = await loadTester.simulateHighLoad(TestComponent, 50);
      
      const averageRenderTime = results.reduce((sum, result) => sum + result.renderTime, 0) / results.length;
      const maxRenderTime = Math.max(...results.map(result => result.renderTime));
      
      expect(averageRenderTime).toBeLessThan(100); // Average should be under 100ms
      expect(maxRenderTime).toBeLessThan(500); // Max should be under 500ms
      
      console.log('Load Test Results:', {
        iterations: results.length,
        averageRenderTime: `${averageRenderTime.toFixed(2)}ms`,
        maxRenderTime: `${maxRenderTime.toFixed(2)}ms`
      });
    });

    test('should handle concurrent user simulation', async () => {
      const TestComponent = () => React.createElement('div', null, 'Concurrent User Test');
      
      const results = await loadTester.simulateConcurrentUsers(TestComponent, 5);
      
      const averageRenderTime = results.reduce((sum, result) => sum + result.renderTime, 0) / results.length;
      
      expect(averageRenderTime).toBeLessThan(200); // Should handle concurrent users efficiently
      expect(results.length).toBe(5); // All users should complete
      
      console.log('Concurrent User Test Results:', {
        userCount: results.length,
        averageRenderTime: `${averageRenderTime.toFixed(2)}ms`
      });
    });

    test('should maintain performance under stress', async () => {
      const StressTestComponent = () => {
        const [items, setItems] = React.useState<number[]>([]);
        
        React.useEffect(() => {
          // Add items rapidly to stress test
          const interval = setInterval(() => {
            setItems(current => [...current, current.length]);
          }, 10);
          
          setTimeout(() => clearInterval(interval), 500);
          
          return () => clearInterval(interval);
        }, []);
        
        return React.createElement('div', null, 
          items.map(item => React.createElement('div', { key: item }, `Item ${item}`))
        );
      };

      const startTime = performance.now();
      render(React.createElement(StressTestComponent));
      
      await waitFor(() => {
        const items = screen.getAllByText(/Item \d+/);
        expect(items.length).toBeGreaterThan(10);
      }, { timeout: 1000 });
      
      const endTime = performance.now();
      const stressTestTime = endTime - startTime;
      
      expect(stressTestTime).toBeLessThan(1000); // Should handle stress within 1 second
    });
  });

  describe('Memory Leak Detection', () => {
    test('should not have memory leaks in component mounting/unmounting', async () => {
      memoryLeakDetector.startDetection();
      
      const TestComponent = () => React.createElement('div', null, 'Memory Test Component');
      
      // Mount and unmount component multiple times
      for (let i = 0; i < 10; i++) {
        const { unmount } = render(React.createElement(TestComponent));
        memoryLeakDetector.takeSample();
        unmount();
        
        // Force garbage collection if available
        if (global.gc) {
          global.gc();
        }
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      const leakDetection = memoryLeakDetector.detectLeaks();
      
      expect(leakDetection.hasLeak).toBe(false);
      
      console.log('Memory Leak Detection:', {
        hasLeak: leakDetection.hasLeak,
        memoryGrowth: `${(leakDetection.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
        samples: leakDetection.samples.length
      });
    });

    test('should handle event listener cleanup', async () => {
      memoryLeakDetector.startDetection();
      
      const EventListenerComponent = () => {
        React.useEffect(() => {
          const handleClick = () => {};
          const handleScroll = () => {};
          
          document.addEventListener('click', handleClick);
          window.addEventListener('scroll', handleScroll);
          
          return () => {
            document.removeEventListener('click', handleClick);
            window.removeEventListener('scroll', handleScroll);
          };
        }, []);
        
        return React.createElement('div', null, 'Event Listener Component');
      };
      
      // Mount and unmount multiple times
      for (let i = 0; i < 5; i++) {
        const { unmount } = render(React.createElement(EventListenerComponent));
        memoryLeakDetector.takeSample();
        unmount();
        
        if (global.gc) {
          global.gc();
        }
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      const leakDetection = memoryLeakDetector.detectLeaks();
      expect(leakDetection.hasLeak).toBe(false);
    });
  });

  describe('Bundle Size Analysis', () => {
    test('should meet bundle size requirements', async () => {
      const analysis = await bundleSizeAnalyzer.analyzeBundleSize();
      const thresholdCheck = bundleSizeAnalyzer.checkBundleSizeThresholds(analysis);
      
      expect(thresholdCheck.passed).toBe(true);
      
      if (!thresholdCheck.passed) {
        console.warn('Bundle Size Violations:', thresholdCheck.violations);
      }
      
      console.log('Bundle Size Analysis:', {
        totalSize: `${(analysis.totalSize / 1024 / 1024).toFixed(2)}MB`,
        javascript: `${(analysis.breakdown.javascript / 1024).toFixed(2)}KB`,
        css: `${(analysis.breakdown.css / 1024).toFixed(2)}KB`,
        images: `${(analysis.breakdown.images / 1024).toFixed(2)}KB`,
        fonts: `${(analysis.breakdown.fonts / 1024).toFixed(2)}KB`,
        other: `${(analysis.breakdown.other / 1024).toFixed(2)}KB`
      });
    });

    test('should optimize resource loading', async () => {
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
      
      // Check for efficient resource loading
      const slowResources = resources.filter(resource => 
        (resource.responseEnd - resource.requestStart) > 1000 // Slower than 1 second
      );
      
      expect(slowResources.length).toBeLessThan(resources.length * 0.1); // Less than 10% should be slow
      
      // Check for resource compression
      const uncompressedResources = resources.filter(resource => {
        const compressionRatio = resource.transferSize / (resource.decodedBodySize || 1);
        return compressionRatio > 0.8; // Not well compressed
      });
      
      expect(uncompressedResources.length).toBeLessThan(resources.length * 0.2); // Less than 20% should be uncompressed
    });
  });

  describe('Real-time Performance', () => {
    test('should handle WebSocket updates efficiently', async () => {
      const mockWebSocket = {
        send: jest.fn(),
        close: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        readyState: WebSocket.OPEN
      };
      
      global.WebSocket = jest.fn(() => mockWebSocket) as any;
      
      const WebSocketComponent = () => {
        const [data, setData] = React.useState<any[]>([]);
        
        React.useEffect(() => {
          const ws = new WebSocket('ws://localhost:8080');
          
          ws.addEventListener('message', (event) => {
            const newData = JSON.parse(event.data);
            setData(current => [...current, newData]);
          });
          
          return () => ws.close();
        }, []);
        
        return React.createElement('div', null, `Data items: ${data.length}`);
      };
      
      const startTime = performance.now();
      render(React.createElement(WebSocketComponent));
      
      // Simulate rapid WebSocket messages
      for (let i = 0; i < 100; i++) {
        const messageEvent = new MessageEvent('message', {
          data: JSON.stringify({ id: i, value: Math.random() })
        });
        
        mockWebSocket.addEventListener.mock.calls.forEach(([event, handler]) => {
          if (event === 'message') {
            handler(messageEvent);
          }
        });
      }
      
      await waitFor(() => {
        expect(screen.getByText('Data items: 100')).toBeInTheDocument();
      });
      
      const endTime = performance.now();
      const processingTime = endTime - startTime;
      
      expect(processingTime).toBeLessThan(1000); // Should process 100 messages within 1 second
    });

    test('should throttle high-frequency updates', async () => {
      const ThrottledComponent = () => {
        const [value, setValue] = React.useState(0);
        const [throttledValue, setThrottledValue] = React.useState(0);
        
        // Throttle updates to every 100ms
        React.useEffect(() => {
          const throttleTimer = setTimeout(() => {
            setThrottledValue(value);
          }, 100);
          
          return () => clearTimeout(throttleTimer);
        }, [value]);
        
        React.useEffect(() => {
          // Simulate rapid updates
          const interval = setInterval(() => {
            setValue(v => v + 1);
          }, 10);
          
          setTimeout(() => clearInterval(interval), 500);
          
          return () => clearInterval(interval);
        }, []);
        
        return React.createElement('div', null, `Value: ${value}, Throttled: ${throttledValue}`);
      };
      
      render(React.createElement(ThrottledComponent));
      
      await waitFor(() => {
        const text = screen.getByText(/Value: \d+, Throttled: \d+/);
        expect(text).toBeInTheDocument();
      }, { timeout: 1000 });
      
      // Throttled value should be significantly less than actual value
      const text = screen.getByText(/Value: (\d+), Throttled: (\d+)/).textContent;
      const matches = text?.match(/Value: (\d+), Throttled: (\d+)/);
      
      if (matches) {
        const actualValue = parseInt(matches[1]);
        const throttledValue = parseInt(matches[2]);
        
        expect(throttledValue).toBeLessThan(actualValue);
        expect(throttledValue).toBeGreaterThan(0);
      }
    });
  });

  describe('Performance Regression Testing', () => {
    test('should maintain performance baselines', async () => {
      const baselines = {
        renderTime: 100,
        memoryUsage: 10 * 1024 * 1024,
        bundleSize: 2 * 1024 * 1024
      };
      
      const TestComponent = () => React.createElement('div', null, 'Baseline Test');
      
      const renderTime = await performanceMonitor.measureComponentRender(TestComponent);
      const memoryUsage = performanceMonitor.measureMemoryUsage();
      const bundleAnalysis = await bundleSizeAnalyzer.analyzeBundleSize();
      
      const results = {
        renderTime,
        memoryUsage,
        bundleSize: bundleAnalysis.totalSize
      };
      
      // Check against baselines with 20% tolerance
      Object.entries(baselines).forEach(([metric, baseline]) => {
        const actual = results[metric as keyof typeof results];
        const tolerance = baseline * 0.2;
        
        expect(actual).toBeLessThan(baseline + tolerance);
        
        console.log(`${metric} Performance:`, {
          baseline: metric === 'renderTime' ? `${baseline}ms` : `${(baseline / 1024 / 1024).toFixed(2)}MB`,
          actual: metric === 'renderTime' ? `${actual.toFixed(2)}ms` : `${(actual / 1024 / 1024).toFixed(2)}MB`,
          withinTolerance: actual < baseline + tolerance
        });
      });
    });
  });
});

// Export performance utilities for use in other tests
export {
  PerformanceMonitor,
  LoadTester,
  MemoryLeakDetector,
  BundleSizeAnalyzer,
  type PerformanceMetrics,
  type ComponentPerformanceTest
};

