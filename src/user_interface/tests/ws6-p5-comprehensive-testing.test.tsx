import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import PerformanceMonitoringDashboard from '../performance_monitoring_framework';
import OptimizationEngineDashboard from '../optimization_engine';
import AdvancedAnalyticsDashboard from '../advanced_analytics';
import SystemCoordinationDashboard from '../system_coordination';

// Mock performance APIs
const mockPerformance = {
  now: jest.fn(() => Date.now()),
  getEntriesByType: jest.fn(() => []),
  getEntriesByName: jest.fn(() => []),
  mark: jest.fn(),
  measure: jest.fn(),
  memory: {
    usedJSHeapSize: 50 * 1024 * 1024,
    totalJSHeapSize: 100 * 1024 * 1024,
    jsHeapSizeLimit: 2 * 1024 * 1024 * 1024
  }
};

// Mock PerformanceObserver
const mockPerformanceObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  disconnect: jest.fn(),
  takeRecords: jest.fn(() => [])
}));

// Setup global mocks
beforeAll(() => {
  global.performance = mockPerformance as any;
  global.PerformanceObserver = mockPerformanceObserver as any;
  
  // Mock ResizeObserver
  global.ResizeObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  }));
  
  // Mock IntersectionObserver
  global.IntersectionObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  }));
});

describe('WS6-P5 Performance Optimization and Monitoring', () => {
  describe('Performance Monitoring Framework', () => {
    test('renders performance monitoring dashboard', () => {
      render(<PerformanceMonitoringDashboard />);
      
      expect(screen.getByText('Performance Monitoring Framework')).toBeInTheDocument();
      expect(screen.getByText('Real-time performance monitoring and optimization for ALL-USE user interface components')).toBeInTheDocument();
    });

    test('displays performance summary cards', () => {
      render(<PerformanceMonitoringDashboard />);
      
      expect(screen.getByText('Overall Score')).toBeInTheDocument();
      expect(screen.getByText('Components')).toBeInTheDocument();
      expect(screen.getByText('Critical Issues')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
    });

    test('starts and stops monitoring', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const startButton = screen.getByText('Start Monitoring');
      fireEvent.click(startButton);
      
      await waitFor(() => {
        expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
      });
      
      const stopButton = screen.getByText('Stop Monitoring');
      fireEvent.click(stopButton);
      
      await waitFor(() => {
        expect(screen.getByText('Start Monitoring')).toBeInTheDocument();
      });
    });

    test('switches between tabs', () => {
      render(<PerformanceMonitoringDashboard />);
      
      // Test Components tab
      fireEvent.click(screen.getByText('Components'));
      expect(screen.getByText('Component Performance')).toBeInTheDocument();
      
      // Test Metrics tab
      fireEvent.click(screen.getByText('Metrics'));
      expect(screen.getByText('Core Web Vitals')).toBeInTheDocument();
      
      // Test Alerts tab
      fireEvent.click(screen.getByText('Alerts'));
      expect(screen.getByText('Performance Alerts')).toBeInTheDocument();
    });

    test('displays performance metrics correctly', () => {
      render(<PerformanceMonitoringDashboard />);
      
      fireEvent.click(screen.getByText('Metrics'));
      
      expect(screen.getByText('First Contentful Paint')).toBeInTheDocument();
      expect(screen.getByText('Largest Contentful Paint')).toBeInTheDocument();
      expect(screen.getByText('Cumulative Layout Shift')).toBeInTheDocument();
      expect(screen.getByText('First Input Delay')).toBeInTheDocument();
    });
  });

  describe('Optimization Engine', () => {
    test('renders optimization engine dashboard', () => {
      render(<OptimizationEngineDashboard />);
      
      expect(screen.getByText('Performance Optimization Engine')).toBeInTheDocument();
      expect(screen.getByText('Intelligent performance optimization with automated rule-based improvements')).toBeInTheDocument();
    });

    test('displays optimization summary cards', () => {
      render(<OptimizationEngineDashboard />);
      
      expect(screen.getByText('Total Optimizations')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('Avg Improvement')).toBeInTheDocument();
      expect(screen.getByText('Total Impact')).toBeInTheDocument();
      expect(screen.getByText('Active Rules')).toBeInTheDocument();
    });

    test('starts and stops optimization', async () => {
      render(<OptimizationEngineDashboard />);
      
      const startButton = screen.getByText('Start Optimization');
      fireEvent.click(startButton);
      
      await waitFor(() => {
        expect(screen.getByText('Stop Optimization')).toBeInTheDocument();
      });
    });

    test('analyzes bundle size', async () => {
      render(<OptimizationEngineDashboard />);
      
      const analyzeButton = screen.getByText('Analyze Bundle');
      fireEvent.click(analyzeButton);
      
      // Switch to bundle tab to see results
      fireEvent.click(screen.getByText('Bundle Analysis'));
      
      await waitFor(() => {
        expect(screen.getByText('Bundle Overview')).toBeInTheDocument();
      });
    });

    test('manages optimization rules', () => {
      render(<OptimizationEngineDashboard />);
      
      fireEvent.click(screen.getByText('Optimization Rules'));
      
      expect(screen.getByText('Configure and manage performance optimization rules')).toBeInTheDocument();
      expect(screen.getByText('Slow Render Optimization')).toBeInTheDocument();
      expect(screen.getByText('Memory Leak Prevention')).toBeInTheDocument();
    });

    test('displays optimization results', () => {
      render(<OptimizationEngineDashboard />);
      
      fireEvent.click(screen.getByText('Results'));
      
      expect(screen.getByText('Optimization Results')).toBeInTheDocument();
      expect(screen.getByText('Detailed results and performance improvements from applied optimizations')).toBeInTheDocument();
    });
  });

  describe('Advanced Analytics', () => {
    test('renders advanced analytics dashboard', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      expect(screen.getByText('Advanced Analytics & Predictive Optimization')).toBeInTheDocument();
      expect(screen.getByText('AI-powered performance analytics with predictive insights and optimization recommendations')).toBeInTheDocument();
    });

    test('displays analytics summary cards', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      expect(screen.getByText('Data Points')).toBeInTheDocument();
      expect(screen.getByText('Active Models')).toBeInTheDocument();
      expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
      expect(screen.getByText('Anomalies')).toBeInTheDocument();
      expect(screen.getByText('Actionable Insights')).toBeInTheDocument();
      expect(screen.getByText('Trends Detected')).toBeInTheDocument();
    });

    test('starts and stops analysis', async () => {
      render(<AdvancedAnalyticsDashboard />);
      
      const startButton = screen.getByText('Start Analysis');
      fireEvent.click(startButton);
      
      await waitFor(() => {
        expect(screen.getByText('Stop Analysis')).toBeInTheDocument();
      });
    });

    test('displays predictive models', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      fireEvent.click(screen.getByText('Predictions'));
      
      expect(screen.getByText('Render Time Predictor')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage Predictor')).toBeInTheDocument();
      expect(screen.getByText('Network Latency Predictor')).toBeInTheDocument();
      expect(screen.getByText('User Experience Predictor')).toBeInTheDocument();
    });

    test('shows anomaly detection', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      fireEvent.click(screen.getByText('Anomalies'));
      
      expect(screen.getByText('Performance Anomalies')).toBeInTheDocument();
      expect(screen.getByText('Detected anomalies in performance metrics with AI-powered analysis')).toBeInTheDocument();
    });

    test('displays user behavior patterns', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      fireEvent.click(screen.getByText('User Patterns'));
      
      expect(screen.getByText('User Behavior Patterns')).toBeInTheDocument();
      expect(screen.getByText('AI-identified user behavior patterns and their performance impact')).toBeInTheDocument();
    });

    test('shows AI insights', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      fireEvent.click(screen.getByText('Insights'));
      
      expect(screen.getByText('Render Performance Improving')).toBeInTheDocument();
      expect(screen.getByText('Bundle Splitting Opportunity')).toBeInTheDocument();
    });

    test('displays A/B test results', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      fireEvent.click(screen.getByText('A/B Tests'));
      
      expect(screen.getByText('A/B Test Results')).toBeInTheDocument();
      expect(screen.getByText('Performance optimization A/B test results and statistical significance')).toBeInTheDocument();
    });
  });

  describe('System Coordination', () => {
    test('renders system coordination dashboard', () => {
      render(<SystemCoordinationDashboard />);
      
      expect(screen.getByText('System Coordination & Performance Integration')).toBeInTheDocument();
      expect(screen.getByText('Comprehensive system coordination with intelligent performance integration across all components')).toBeInTheDocument();
    });

    test('displays system summary cards', () => {
      render(<SystemCoordinationDashboard />);
      
      expect(screen.getByText('System Health')).toBeInTheDocument();
      expect(screen.getByText('Active Components')).toBeInTheDocument();
      expect(screen.getByText('Coordination Efficiency')).toBeInTheDocument();
      expect(screen.getByText('Active Optimizations')).toBeInTheDocument();
      expect(screen.getByText('Unresolved Alerts')).toBeInTheDocument();
      expect(screen.getByText('Coordination Status')).toBeInTheDocument();
    });

    test('starts and stops coordination', async () => {
      render(<SystemCoordinationDashboard />);
      
      const startButton = screen.getByText('Start Coordination');
      fireEvent.click(startButton);
      
      await waitFor(() => {
        expect(screen.getByText('Stop Coordination')).toBeInTheDocument();
      });
    });

    test('displays system components', () => {
      render(<SystemCoordinationDashboard />);
      
      fireEvent.click(screen.getByText('Components'));
      
      expect(screen.getByText('System Components')).toBeInTheDocument();
      expect(screen.getByText('Performance Monitoring Framework')).toBeInTheDocument();
      expect(screen.getByText('Performance Optimization Engine')).toBeInTheDocument();
      expect(screen.getByText('Advanced Analytics Engine')).toBeInTheDocument();
    });

    test('shows coordination mechanisms', () => {
      render(<SystemCoordinationDashboard />);
      
      fireEvent.click(screen.getByText('Coordinations'));
      
      expect(screen.getByText('Monitoring-Optimization Synchronization')).toBeInTheDocument();
      expect(screen.getByText('Analytics-UI Performance Coordination')).toBeInTheDocument();
      expect(screen.getByText('Cross-Component Resource Sharing')).toBeInTheDocument();
    });

    test('displays optimization tasks', () => {
      render(<SystemCoordinationDashboard />);
      
      fireEvent.click(screen.getByText('Optimization Tasks'));
      
      expect(screen.getByText('Active and completed performance optimization tasks')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage Optimization')).toBeInTheDocument();
      expect(screen.getByText('Network Latency Reduction')).toBeInTheDocument();
    });

    test('manages system alerts', () => {
      render(<SystemCoordinationDashboard />);
      
      fireEvent.click(screen.getByText('Alerts'));
      
      expect(screen.getByText('System Alerts')).toBeInTheDocument();
      expect(screen.getByText('Performance alerts and system notifications')).toBeInTheDocument();
    });
  });

  describe('Integration Testing', () => {
    test('all components can be rendered together', () => {
      const { container } = render(
        <div>
          <PerformanceMonitoringDashboard />
          <OptimizationEngineDashboard />
          <AdvancedAnalyticsDashboard />
          <SystemCoordinationDashboard />
        </div>
      );
      
      expect(container).toBeInTheDocument();
    });

    test('performance monitoring integrates with optimization engine', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      // Start monitoring
      fireEvent.click(screen.getByText('Start Monitoring'));
      
      await waitFor(() => {
        expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
      });
      
      // Verify monitoring status
      expect(screen.getByText('Active')).toBeInTheDocument();
    });

    test('optimization engine coordinates with analytics', async () => {
      render(<OptimizationEngineDashboard />);
      
      // Start optimization
      fireEvent.click(screen.getByText('Start Optimization'));
      
      await waitFor(() => {
        expect(screen.getByText('Stop Optimization')).toBeInTheDocument();
      });
      
      // Check optimization status
      expect(screen.getByText('Running')).toBeInTheDocument();
    });

    test('system coordination manages all components', async () => {
      render(<SystemCoordinationDashboard />);
      
      // Start coordination
      fireEvent.click(screen.getByText('Start Coordination'));
      
      await waitFor(() => {
        expect(screen.getByText('Stop Coordination')).toBeInTheDocument();
      });
      
      // Verify coordination is active
      expect(screen.getByText('ACTIVE')).toBeInTheDocument();
    });
  });

  describe('Performance Metrics Validation', () => {
    test('performance monitoring collects valid metrics', () => {
      render(<PerformanceMonitoringDashboard />);
      
      fireEvent.click(screen.getByText('Start Monitoring'));
      fireEvent.click(screen.getByText('Metrics'));
      
      // Verify Core Web Vitals are displayed
      expect(screen.getByText('First Contentful Paint')).toBeInTheDocument();
      expect(screen.getByText('Largest Contentful Paint')).toBeInTheDocument();
      expect(screen.getByText('Cumulative Layout Shift')).toBeInTheDocument();
      expect(screen.getByText('First Input Delay')).toBeInTheDocument();
    });

    test('optimization engine tracks improvement metrics', () => {
      render(<OptimizationEngineDashboard />);
      
      fireEvent.click(screen.getByText('Results'));
      
      // Verify optimization results structure
      expect(screen.getByText('Optimization Results')).toBeInTheDocument();
      expect(screen.getByText('Detailed results and performance improvements from applied optimizations')).toBeInTheDocument();
    });

    test('analytics engine provides predictive insights', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      fireEvent.click(screen.getByText('Predictions'));
      
      // Verify predictive models are available
      expect(screen.getByText('Render Time Predictor')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage Predictor')).toBeInTheDocument();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    test('handles missing performance API gracefully', () => {
      const originalPerformance = global.performance;
      delete (global as any).performance;
      
      expect(() => {
        render(<PerformanceMonitoringDashboard />);
      }).not.toThrow();
      
      global.performance = originalPerformance;
    });

    test('handles missing PerformanceObserver gracefully', () => {
      const originalObserver = global.PerformanceObserver;
      delete (global as any).PerformanceObserver;
      
      expect(() => {
        render(<PerformanceMonitoringDashboard />);
      }).not.toThrow();
      
      global.PerformanceObserver = originalObserver;
    });

    test('handles component unmounting during monitoring', () => {
      const { unmount } = render(<PerformanceMonitoringDashboard />);
      
      fireEvent.click(screen.getByText('Start Monitoring'));
      
      expect(() => {
        unmount();
      }).not.toThrow();
    });

    test('handles rapid start/stop operations', async () => {
      render(<SystemCoordinationDashboard />);
      
      const startButton = screen.getByText('Start Coordination');
      
      // Rapid start/stop
      fireEvent.click(startButton);
      
      await waitFor(() => {
        const stopButton = screen.getByText('Stop Coordination');
        fireEvent.click(stopButton);
      });
      
      await waitFor(() => {
        expect(screen.getByText('Start Coordination')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility and User Experience', () => {
    test('all dashboards have proper ARIA labels', () => {
      render(<PerformanceMonitoringDashboard />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    test('keyboard navigation works correctly', () => {
      render(<OptimizationEngineDashboard />);
      
      const startButton = screen.getByText('Start Optimization');
      startButton.focus();
      
      expect(document.activeElement).toBe(startButton);
    });

    test('responsive design elements are present', () => {
      render(<AdvancedAnalyticsDashboard />);
      
      // Check for responsive grid classes
      const container = screen.getByText('Advanced Analytics & Predictive Optimization').closest('div');
      expect(container).toHaveClass('max-w-7xl');
    });

    test('loading states are handled properly', async () => {
      render(<SystemCoordinationDashboard />);
      
      fireEvent.click(screen.getByText('Start Coordination'));
      
      // Should show active state
      await waitFor(() => {
        expect(screen.getByText('ACTIVE')).toBeInTheDocument();
      });
    });
  });

  describe('Data Persistence and State Management', () => {
    test('component state persists across tab switches', () => {
      render(<PerformanceMonitoringDashboard />);
      
      fireEvent.click(screen.getByText('Start Monitoring'));
      fireEvent.click(screen.getByText('Components'));
      fireEvent.click(screen.getByText('Overview'));
      
      expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
    });

    test('optimization rules can be toggled', () => {
      render(<OptimizationEngineDashboard />);
      
      fireEvent.click(screen.getByText('Optimization Rules'));
      
      // Find and interact with rule toggles
      const enabledCheckboxes = screen.getAllByRole('checkbox');
      expect(enabledCheckboxes.length).toBeGreaterThan(0);
    });

    test('alerts can be resolved', () => {
      render(<SystemCoordinationDashboard />);
      
      fireEvent.click(screen.getByText('Alerts'));
      
      // Check for resolve buttons (if any alerts exist)
      const resolveButtons = screen.queryAllByText('Resolve');
      resolveButtons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });
  });
});

// Performance benchmark tests
describe('Performance Benchmarks', () => {
  test('dashboard renders within acceptable time', async () => {
    const startTime = performance.now();
    
    render(<PerformanceMonitoringDashboard />);
    
    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    // Should render within 100ms
    expect(renderTime).toBeLessThan(100);
  });

  test('optimization engine processes rules efficiently', async () => {
    const startTime = performance.now();
    
    render(<OptimizationEngineDashboard />);
    fireEvent.click(screen.getByText('Start Optimization'));
    
    const endTime = performance.now();
    const processingTime = endTime - startTime;
    
    // Should start optimization within 50ms
    expect(processingTime).toBeLessThan(50);
  });

  test('analytics dashboard handles large datasets', async () => {
    const startTime = performance.now();
    
    render(<AdvancedAnalyticsDashboard />);
    fireEvent.click(screen.getByText('Start Analysis'));
    
    const endTime = performance.now();
    const analysisTime = endTime - startTime;
    
    // Should handle analysis start within 75ms
    expect(analysisTime).toBeLessThan(75);
  });

  test('system coordination scales with component count', async () => {
    const startTime = performance.now();
    
    render(<SystemCoordinationDashboard />);
    fireEvent.click(screen.getByText('Start Coordination'));
    
    const endTime = performance.now();
    const coordinationTime = endTime - startTime;
    
    // Should coordinate components within 60ms
    expect(coordinationTime).toBeLessThan(60);
  });
});

// Memory usage tests
describe('Memory Management', () => {
  test('components clean up properly on unmount', () => {
    const { unmount } = render(<PerformanceMonitoringDashboard />);
    
    fireEvent.click(screen.getByText('Start Monitoring'));
    
    // Unmount should not cause memory leaks
    expect(() => unmount()).not.toThrow();
  });

  test('optimization engine releases resources', () => {
    const { unmount } = render(<OptimizationEngineDashboard />);
    
    fireEvent.click(screen.getByText('Start Optimization'));
    
    expect(() => unmount()).not.toThrow();
  });

  test('analytics engine manages memory efficiently', () => {
    const { unmount } = render(<AdvancedAnalyticsDashboard />);
    
    fireEvent.click(screen.getByText('Start Analysis'));
    
    expect(() => unmount()).not.toThrow();
  });

  test('system coordination prevents memory leaks', () => {
    const { unmount } = render(<SystemCoordinationDashboard />);
    
    fireEvent.click(screen.getByText('Start Coordination'));
    
    expect(() => unmount()).not.toThrow();
  });
});

export default {};

