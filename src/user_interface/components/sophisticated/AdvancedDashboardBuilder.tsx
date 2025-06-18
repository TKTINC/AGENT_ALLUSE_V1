import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, useSpring, useMotionValue, useTransform } from 'framer-motion';
import { 
  Layers, 
  Grid, 
  BarChart3, 
  TrendingUp, 
  Settings, 
  Maximize2, 
  Minimize2,
  RotateCcw,
  Save,
  Download,
  Share2,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Zap,
  Target,
  Compass,
  Activity
} from 'lucide-react';

// Advanced Interface Components for WS6-P3
// Sophisticated user interactions with professional animations and enterprise-grade functionality

// Advanced Dashboard Builder Component
interface DashboardBuilderProps {
  initialLayout?: DashboardLayout;
  onLayoutChange?: (layout: DashboardLayout) => void;
  onSave?: (layout: DashboardLayout) => void;
  availableWidgets?: WidgetDefinition[];
  isEditing?: boolean;
}

interface DashboardLayout {
  id: string;
  name: string;
  widgets: WidgetInstance[];
  layout: GridLayout[];
  theme: 'light' | 'dark' | 'auto';
  settings: DashboardSettings;
}

interface WidgetInstance {
  id: string;
  type: string;
  title: string;
  config: Record<string, any>;
  position: { x: number; y: number; w: number; h: number };
  isLocked?: boolean;
  isVisible?: boolean;
}

interface WidgetDefinition {
  type: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  defaultSize: { w: number; h: number };
  configSchema: any;
  component: React.ComponentType<any>;
}

interface GridLayout {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
  static?: boolean;
}

interface DashboardSettings {
  gridSize: number;
  margin: number;
  autoSave: boolean;
  snapToGrid: boolean;
  showGrid: boolean;
  allowOverlap: boolean;
}

export const AdvancedDashboardBuilder: React.FC<DashboardBuilderProps> = ({
  initialLayout,
  onLayoutChange,
  onSave,
  availableWidgets = [],
  isEditing = false
}) => {
  const [layout, setLayout] = useState<DashboardLayout>(
    initialLayout || createDefaultLayout()
  );
  const [selectedWidget, setSelectedWidget] = useState<string | null>(null);
  const [draggedWidget, setDraggedWidget] = useState<WidgetDefinition | null>(null);
  const [isGridVisible, setIsGridVisible] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const lastPanPoint = useRef({ x: 0, y: 0 });

  // Advanced gesture handling
  const handlePanStart = useCallback((event: React.MouseEvent) => {
    if (event.button === 1 || (event.button === 0 && event.ctrlKey)) {
      isDragging.current = true;
      lastPanPoint.current = { x: event.clientX, y: event.clientY };
      event.preventDefault();
    }
  }, []);

  const handlePanMove = useCallback((event: React.MouseEvent) => {
    if (isDragging.current) {
      const deltaX = event.clientX - lastPanPoint.current.x;
      const deltaY = event.clientY - lastPanPoint.current.y;
      
      setPanOffset(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      
      lastPanPoint.current = { x: event.clientX, y: event.clientY };
    }
  }, []);

  const handlePanEnd = useCallback(() => {
    isDragging.current = false;
  }, []);

  // Zoom handling with wheel
  const handleWheel = useCallback((event: React.WheelEvent) => {
    if (event.ctrlKey) {
      event.preventDefault();
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      setZoom(prev => Math.max(0.1, Math.min(3, prev * delta)));
    }
  }, []);

  // Widget drag and drop
  const handleWidgetDragStart = useCallback((widget: WidgetDefinition) => {
    setDraggedWidget(widget);
  }, []);

  const handleWidgetDrop = useCallback((event: React.DragEvent, position: { x: number; y: number }) => {
    if (draggedWidget) {
      const newWidget: WidgetInstance = {
        id: `widget-${Date.now()}`,
        type: draggedWidget.type,
        title: draggedWidget.name,
        config: {},
        position: {
          x: Math.round(position.x / layout.settings.gridSize),
          y: Math.round(position.y / layout.settings.gridSize),
          w: draggedWidget.defaultSize.w,
          h: draggedWidget.defaultSize.h
        }
      };

      setLayout(prev => ({
        ...prev,
        widgets: [...prev.widgets, newWidget]
      }));

      setDraggedWidget(null);
    }
  }, [draggedWidget, layout.settings.gridSize]);

  // Widget manipulation
  const updateWidget = useCallback((widgetId: string, updates: Partial<WidgetInstance>) => {
    setLayout(prev => ({
      ...prev,
      widgets: prev.widgets.map(widget =>
        widget.id === widgetId ? { ...widget, ...updates } : widget
      )
    }));
  }, []);

  const deleteWidget = useCallback((widgetId: string) => {
    setLayout(prev => ({
      ...prev,
      widgets: prev.widgets.filter(widget => widget.id !== widgetId)
    }));
  }, []);

  const duplicateWidget = useCallback((widgetId: string) => {
    const widget = layout.widgets.find(w => w.id === widgetId);
    if (widget) {
      const newWidget: WidgetInstance = {
        ...widget,
        id: `widget-${Date.now()}`,
        position: {
          ...widget.position,
          x: widget.position.x + 1,
          y: widget.position.y + 1
        }
      };
      
      setLayout(prev => ({
        ...prev,
        widgets: [...prev.widgets, newWidget]
      }));
    }
  }, [layout.widgets]);

  // Layout operations
  const saveLayout = useCallback(() => {
    onSave?.(layout);
  }, [layout, onSave]);

  const resetLayout = useCallback(() => {
    setLayout(createDefaultLayout());
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  }, []);

  const exportLayout = useCallback(() => {
    const dataStr = JSON.stringify(layout, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${layout.name}-layout.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [layout]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 's':
            event.preventDefault();
            saveLayout();
            break;
          case 'z':
            event.preventDefault();
            // Implement undo functionality
            break;
          case 'y':
            event.preventDefault();
            // Implement redo functionality
            break;
          case 'g':
            event.preventDefault();
            setIsGridVisible(prev => !prev);
            break;
          case '0':
            event.preventDefault();
            setZoom(1);
            setPanOffset({ x: 0, y: 0 });
            break;
        }
      }
      
      if (event.key === 'Delete' && selectedWidget) {
        deleteWidget(selectedWidget);
        setSelectedWidget(null);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [saveLayout, selectedWidget, deleteWidget]);

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Dashboard Builder
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsGridVisible(!isGridVisible)}
              className={`p-2 rounded-lg transition-colors ${
                isGridVisible 
                  ? 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400'
                  : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
              }`}
              title="Toggle Grid (Ctrl+G)"
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={resetLayout}
              className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600 transition-colors"
              title="Reset Layout"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-lg">
            <span className="text-sm text-gray-600 dark:text-gray-400">Zoom:</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {Math.round(zoom * 100)}%
            </span>
          </div>
          <button
            onClick={saveLayout}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
            title="Save Layout (Ctrl+S)"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          <button
            onClick={exportLayout}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
            title="Export Layout"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Widget Palette */}
        <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 p-4 overflow-y-auto">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-4">
            Available Widgets
          </h3>
          <div className="space-y-2">
            {availableWidgets.map((widget) => (
              <motion.div
                key={widget.type}
                draggable
                onDragStart={() => handleWidgetDragStart(widget)}
                className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg cursor-move hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center gap-3">
                  <div className="text-blue-600 dark:text-blue-400">
                    {widget.icon}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {widget.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {widget.description}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 relative overflow-hidden">
          <div
            ref={containerRef}
            className="w-full h-full relative cursor-grab active:cursor-grabbing"
            onMouseDown={handlePanStart}
            onMouseMove={handlePanMove}
            onMouseUp={handlePanEnd}
            onWheel={handleWheel}
            style={{
              transform: `scale(${zoom}) translate(${panOffset.x}px, ${panOffset.y}px)`,
              transformOrigin: '0 0'
            }}
          >
            {/* Grid Background */}
            {isGridVisible && (
              <div
                className="absolute inset-0 opacity-20"
                style={{
                  backgroundImage: `
                    linear-gradient(to right, #e5e7eb 1px, transparent 1px),
                    linear-gradient(to bottom, #e5e7eb 1px, transparent 1px)
                  `,
                  backgroundSize: `${layout.settings.gridSize}px ${layout.settings.gridSize}px`
                }}
              />
            )}

            {/* Widgets */}
            <AnimatePresence>
              {layout.widgets.map((widget) => (
                <AdvancedWidget
                  key={widget.id}
                  widget={widget}
                  isSelected={selectedWidget === widget.id}
                  onSelect={() => setSelectedWidget(widget.id)}
                  onUpdate={(updates) => updateWidget(widget.id, updates)}
                  onDelete={() => deleteWidget(widget.id)}
                  onDuplicate={() => duplicateWidget(widget.id)}
                  gridSize={layout.settings.gridSize}
                  snapToGrid={layout.settings.snapToGrid}
                />
              ))}
            </AnimatePresence>

            {/* Drop Zone Indicator */}
            {draggedWidget && (
              <div className="absolute inset-0 bg-blue-500 bg-opacity-10 border-2 border-dashed border-blue-500 rounded-lg" />
            )}
          </div>
        </div>

        {/* Properties Panel */}
        {selectedWidget && (
          <WidgetPropertiesPanel
            widget={layout.widgets.find(w => w.id === selectedWidget)!}
            onUpdate={(updates) => updateWidget(selectedWidget, updates)}
            onClose={() => setSelectedWidget(null)}
          />
        )}
      </div>
    </div>
  );
};

// Advanced Widget Component
interface AdvancedWidgetProps {
  widget: WidgetInstance;
  isSelected: boolean;
  onSelect: () => void;
  onUpdate: (updates: Partial<WidgetInstance>) => void;
  onDelete: () => void;
  onDuplicate: () => void;
  gridSize: number;
  snapToGrid: boolean;
}

const AdvancedWidget: React.FC<AdvancedWidgetProps> = ({
  widget,
  isSelected,
  onSelect,
  onUpdate,
  onDelete,
  onDuplicate,
  gridSize,
  snapToGrid
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, w: 0, h: 0 });

  const x = useMotionValue(widget.position.x * gridSize);
  const y = useMotionValue(widget.position.y * gridSize);
  const width = useMotionValue(widget.position.w * gridSize);
  const height = useMotionValue(widget.position.h * gridSize);

  const handleDragStart = useCallback((event: React.MouseEvent) => {
    if (widget.isLocked) return;
    
    setIsDragging(true);
    setDragStart({ x: event.clientX, y: event.clientY });
    onSelect();
  }, [widget.isLocked, onSelect]);

  const handleDrag = useCallback((event: React.MouseEvent) => {
    if (!isDragging || widget.isLocked) return;

    const deltaX = event.clientX - dragStart.x;
    const deltaY = event.clientY - dragStart.y;

    let newX = widget.position.x * gridSize + deltaX;
    let newY = widget.position.y * gridSize + deltaY;

    if (snapToGrid) {
      newX = Math.round(newX / gridSize) * gridSize;
      newY = Math.round(newY / gridSize) * gridSize;
    }

    x.set(newX);
    y.set(newY);
  }, [isDragging, widget.isLocked, widget.position, dragStart, gridSize, snapToGrid, x, y]);

  const handleDragEnd = useCallback(() => {
    if (!isDragging) return;

    setIsDragging(false);
    
    const newPosition = {
      ...widget.position,
      x: Math.round(x.get() / gridSize),
      y: Math.round(y.get() / gridSize)
    };

    onUpdate({ position: newPosition });
  }, [isDragging, widget.position, x, y, gridSize, onUpdate]);

  const handleResizeStart = useCallback((event: React.MouseEvent) => {
    if (widget.isLocked) return;
    
    event.stopPropagation();
    setIsResizing(true);
    setResizeStart({
      x: event.clientX,
      y: event.clientY,
      w: widget.position.w,
      h: widget.position.h
    });
  }, [widget.isLocked, widget.position]);

  const handleResize = useCallback((event: React.MouseEvent) => {
    if (!isResizing || widget.isLocked) return;

    const deltaX = event.clientX - resizeStart.x;
    const deltaY = event.clientY - resizeStart.y;

    let newW = resizeStart.w + Math.round(deltaX / gridSize);
    let newH = resizeStart.h + Math.round(deltaY / gridSize);

    newW = Math.max(1, newW);
    newH = Math.max(1, newH);

    width.set(newW * gridSize);
    height.set(newH * gridSize);
  }, [isResizing, widget.isLocked, resizeStart, gridSize, width, height]);

  const handleResizeEnd = useCallback(() => {
    if (!isResizing) return;

    setIsResizing(false);
    
    const newPosition = {
      ...widget.position,
      w: Math.round(width.get() / gridSize),
      h: Math.round(height.get() / gridSize)
    };

    onUpdate({ position: newPosition });
  }, [isResizing, widget.position, width, height, gridSize, onUpdate]);

  // Context menu
  const handleContextMenu = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    // Implement context menu
  }, []);

  return (
    <motion.div
      className={`absolute bg-white dark:bg-gray-800 rounded-lg shadow-lg border-2 transition-all ${
        isSelected 
          ? 'border-blue-500 shadow-blue-500/20' 
          : 'border-gray-200 dark:border-gray-700'
      } ${widget.isLocked ? 'cursor-not-allowed' : 'cursor-move'}`}
      style={{
        x,
        y,
        width,
        height,
        opacity: widget.isVisible !== false ? 1 : 0.5
      }}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: widget.isVisible !== false ? 1 : 0.5 }}
      exit={{ scale: 0, opacity: 0 }}
      onMouseDown={handleDragStart}
      onMouseMove={isDragging ? handleDrag : isResizing ? handleResize : undefined}
      onMouseUp={isDragging ? handleDragEnd : isResizing ? handleResizeEnd : undefined}
      onContextMenu={handleContextMenu}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Widget Header */}
      <div className="flex items-center justify-between p-2 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <div className="text-blue-600 dark:text-blue-400">
            <BarChart3 className="w-4 h-4" />
          </div>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {widget.title}
          </span>
        </div>
        
        <div className="flex items-center gap-1">
          {widget.isLocked ? (
            <Lock className="w-3 h-3 text-gray-400" />
          ) : (
            <Unlock className="w-3 h-3 text-gray-400" />
          )}
          {widget.isVisible !== false ? (
            <Eye className="w-3 h-3 text-gray-400" />
          ) : (
            <EyeOff className="w-3 h-3 text-gray-400" />
          )}
        </div>
      </div>

      {/* Widget Content */}
      <div className="p-4 h-full">
        <div className="w-full h-full bg-gray-50 dark:bg-gray-700 rounded flex items-center justify-center">
          <span className="text-gray-500 dark:text-gray-400 text-sm">
            {widget.type} Widget
          </span>
        </div>
      </div>

      {/* Resize Handle */}
      {!widget.isLocked && (
        <div
          className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
          onMouseDown={handleResizeStart}
        >
          <div className="absolute bottom-1 right-1 w-2 h-2 bg-blue-500 rounded-sm" />
        </div>
      )}

      {/* Selection Indicators */}
      {isSelected && !widget.isLocked && (
        <>
          <div className="absolute -top-1 -left-1 w-2 h-2 bg-blue-500 rounded-full" />
          <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 rounded-full" />
          <div className="absolute -bottom-1 -left-1 w-2 h-2 bg-blue-500 rounded-full" />
          <div className="absolute -bottom-1 -right-1 w-2 h-2 bg-blue-500 rounded-full" />
        </>
      )}
    </motion.div>
  );
};

// Widget Properties Panel
interface WidgetPropertiesPanelProps {
  widget: WidgetInstance;
  onUpdate: (updates: Partial<WidgetInstance>) => void;
  onClose: () => void;
}

const WidgetPropertiesPanel: React.FC<WidgetPropertiesPanelProps> = ({
  widget,
  onUpdate,
  onClose
}) => {
  return (
    <motion.div
      className="w-80 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 p-4 overflow-y-auto"
      initial={{ x: 320 }}
      animate={{ x: 0 }}
      exit={{ x: 320 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Widget Properties
        </h3>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          ×
        </button>
      </div>

      <div className="space-y-4">
        {/* Basic Properties */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Title
          </label>
          <input
            type="text"
            value={widget.title}
            onChange={(e) => onUpdate({ title: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>

        {/* Position */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              X Position
            </label>
            <input
              type="number"
              value={widget.position.x}
              onChange={(e) => onUpdate({ 
                position: { ...widget.position, x: parseInt(e.target.value) || 0 }
              })}
              className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Y Position
            </label>
            <input
              type="number"
              value={widget.position.y}
              onChange={(e) => onUpdate({ 
                position: { ...widget.position, y: parseInt(e.target.value) || 0 }
              })}
              className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
            />
          </div>
        </div>

        {/* Size */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Width
            </label>
            <input
              type="number"
              value={widget.position.w}
              onChange={(e) => onUpdate({ 
                position: { ...widget.position, w: parseInt(e.target.value) || 1 }
              })}
              className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Height
            </label>
            <input
              type="number"
              value={widget.position.h}
              onChange={(e) => onUpdate({ 
                position: { ...widget.position, h: parseInt(e.target.value) || 1 }
              })}
              className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
            />
          </div>
        </div>

        {/* Visibility and Lock */}
        <div className="space-y-2">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={widget.isVisible !== false}
              onChange={(e) => onUpdate({ isVisible: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Visible</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={widget.isLocked || false}
              onChange={(e) => onUpdate({ isLocked: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Locked</span>
          </label>
        </div>
      </div>
    </motion.div>
  );
};

// Helper function to create default layout
function createDefaultLayout(): DashboardLayout {
  return {
    id: 'default',
    name: 'Default Dashboard',
    widgets: [],
    layout: [],
    theme: 'auto',
    settings: {
      gridSize: 20,
      margin: 10,
      autoSave: true,
      snapToGrid: true,
      showGrid: true,
      allowOverlap: false
    }
  };
}

// Export all components
export {
  AdvancedWidget,
  WidgetPropertiesPanel,
  type DashboardLayout,
  type WidgetInstance,
  type WidgetDefinition
};

