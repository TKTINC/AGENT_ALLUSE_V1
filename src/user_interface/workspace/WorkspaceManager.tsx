// Workspace Management System for WS6-P3
// Advanced workspace creation, management, and collaboration tools

import React, { useState, useEffect, useCallback, useMemo, createContext, useContext } from 'react';
import { 
  Layout, 
  Grid, 
  Columns, 
  Rows, 
  Plus, 
  Edit3, 
  Trash2, 
  Copy, 
  Share2, 
  Download, 
  Upload, 
  Save, 
  RotateCcw, 
  Maximize2, 
  Minimize2, 
  Move, 
  Lock, 
  Unlock, 
  Eye, 
  EyeOff, 
  Settings, 
  Users, 
  Clock, 
  Star, 
  StarOff, 
  Search, 
  Filter, 
  SortAsc, 
  SortDesc, 
  MoreHorizontal,
  Folder,
  FolderOpen,
  Tag,
  Calendar,
  User,
  CheckCircle,
  XCircle,
  AlertCircle,
  Info,
  Zap,
  Target,
  TrendingUp,
  BarChart3,
  PieChart,
  LineChart,
  Activity,
  DollarSign,
  Percent,
  Hash,
  Type,
  Image,
  Video,
  FileText,
  Database,
  Globe,
  Smartphone,
  Tablet,
  Monitor,
  Laptop
} from 'lucide-react';
import { motion, AnimatePresence, useDragControls } from 'framer-motion';

// Workspace Types
export interface WorkspaceWidget {
  id: string;
  type: 'chart' | 'watchlist' | 'orders' | 'positions' | 'news' | 'analytics' | 'calculator' | 'notes' | 'calendar' | 'performance';
  title: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  config: {
    symbol?: string;
    timeframe?: string;
    chartType?: string;
    indicators?: string[];
    theme?: string;
    autoRefresh?: boolean;
    refreshInterval?: number;
    [key: string]: any;
  };
  visible: boolean;
  locked: boolean;
  minimized: boolean;
  zIndex: number;
  createdAt: number;
  updatedAt: number;
}

export interface WorkspaceLayout {
  id: string;
  name: string;
  description: string;
  category: 'trading' | 'analysis' | 'portfolio' | 'research' | 'custom';
  isTemplate: boolean;
  isPublic: boolean;
  isFavorite: boolean;
  tags: string[];
  layout: {
    type: 'grid' | 'freeform' | 'tabs' | 'split';
    gridSize: { columns: number; rows: number };
    snapToGrid: boolean;
    gridSpacing: number;
    backgroundColor: string;
    backgroundImage?: string;
    widgets: WorkspaceWidget[];
  };
  permissions: {
    owner: string;
    editors: string[];
    viewers: string[];
    isPublic: boolean;
  };
  metadata: {
    createdBy: string;
    createdAt: number;
    updatedBy: string;
    updatedAt: number;
    version: number;
    usageCount: number;
    rating: number;
    reviews: WorkspaceReview[];
  };
  settings: {
    autoSave: boolean;
    autoSaveInterval: number;
    showGrid: boolean;
    showRuler: boolean;
    enableCollaboration: boolean;
    allowComments: boolean;
    trackChanges: boolean;
  };
}

export interface WorkspaceReview {
  id: string;
  userId: string;
  userName: string;
  rating: number;
  comment: string;
  timestamp: number;
}

export interface WorkspaceTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  preview: string;
  layout: WorkspaceLayout;
  popularity: number;
  downloads: number;
}

export interface WorkspaceFolder {
  id: string;
  name: string;
  description: string;
  color: string;
  icon: string;
  parentId?: string;
  workspaces: string[];
  subfolders: string[];
  permissions: {
    owner: string;
    editors: string[];
    viewers: string[];
  };
  createdAt: number;
  updatedAt: number;
}

// Widget Types Registry
export const WIDGET_TYPES = {
  chart: {
    name: 'Price Chart',
    description: 'Interactive price chart with technical indicators',
    icon: LineChart,
    defaultSize: { width: 6, height: 4 },
    minSize: { width: 3, height: 2 },
    maxSize: { width: 12, height: 8 },
    configurable: ['symbol', 'timeframe', 'chartType', 'indicators'],
    category: 'trading'
  },
  watchlist: {
    name: 'Watchlist',
    description: 'Monitor multiple symbols and their key metrics',
    icon: Eye,
    defaultSize: { width: 4, height: 6 },
    minSize: { width: 2, height: 3 },
    maxSize: { width: 8, height: 12 },
    configurable: ['symbols', 'columns', 'sorting'],
    category: 'trading'
  },
  orders: {
    name: 'Order Management',
    description: 'View and manage active orders',
    icon: Target,
    defaultSize: { width: 6, height: 4 },
    minSize: { width: 4, height: 3 },
    maxSize: { width: 12, height: 8 },
    configurable: ['filters', 'columns', 'autoRefresh'],
    category: 'trading'
  },
  positions: {
    name: 'Positions',
    description: 'Track open positions and P&L',
    icon: TrendingUp,
    defaultSize: { width: 6, height: 4 },
    minSize: { width: 4, height: 3 },
    maxSize: { width: 12, height: 8 },
    configurable: ['grouping', 'columns', 'calculations'],
    category: 'trading'
  },
  news: {
    name: 'Market News',
    description: 'Latest market news and analysis',
    icon: FileText,
    defaultSize: { width: 4, height: 6 },
    minSize: { width: 3, height: 4 },
    maxSize: { width: 8, height: 12 },
    configurable: ['sources', 'categories', 'keywords'],
    category: 'research'
  },
  analytics: {
    name: 'Analytics Dashboard',
    description: 'Performance metrics and analytics',
    icon: BarChart3,
    defaultSize: { width: 8, height: 6 },
    minSize: { width: 4, height: 4 },
    maxSize: { width: 12, height: 12 },
    configurable: ['metrics', 'timeframe', 'visualization'],
    category: 'analysis'
  },
  calculator: {
    name: 'Position Calculator',
    description: 'Calculate position sizes and risk metrics',
    icon: Hash,
    defaultSize: { width: 3, height: 4 },
    minSize: { width: 2, height: 3 },
    maxSize: { width: 6, height: 8 },
    configurable: ['calculationType', 'defaultValues'],
    category: 'tools'
  },
  notes: {
    name: 'Notes',
    description: 'Trading notes and observations',
    icon: Type,
    defaultSize: { width: 4, height: 4 },
    minSize: { width: 2, height: 2 },
    maxSize: { width: 8, height: 8 },
    configurable: ['formatting', 'sharing', 'tags'],
    category: 'tools'
  },
  calendar: {
    name: 'Economic Calendar',
    description: 'Important economic events and earnings',
    icon: Calendar,
    defaultSize: { width: 6, height: 4 },
    minSize: { width: 4, height: 3 },
    maxSize: { width: 12, height: 8 },
    configurable: ['importance', 'countries', 'categories'],
    category: 'research'
  },
  performance: {
    name: 'Performance Tracker',
    description: 'Track trading performance over time',
    icon: Activity,
    defaultSize: { width: 6, height: 4 },
    minSize: { width: 4, height: 3 },
    maxSize: { width: 12, height: 8 },
    configurable: ['timeframe', 'metrics', 'benchmarks'],
    category: 'analysis'
  }
};

// Workspace Context
interface WorkspaceContextType {
  workspaces: WorkspaceLayout[];
  folders: WorkspaceFolder[];
  activeWorkspace: WorkspaceLayout | null;
  templates: WorkspaceTemplate[];
  isLoading: boolean;
  createWorkspace: (name: string, description: string, category: string) => Promise<WorkspaceLayout>;
  updateWorkspace: (id: string, updates: Partial<WorkspaceLayout>) => Promise<void>;
  deleteWorkspace: (id: string) => Promise<void>;
  duplicateWorkspace: (id: string, newName: string) => Promise<WorkspaceLayout>;
  setActiveWorkspace: (workspace: WorkspaceLayout | null) => void;
  addWidget: (widget: Omit<WorkspaceWidget, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateWidget: (widgetId: string, updates: Partial<WorkspaceWidget>) => void;
  removeWidget: (widgetId: string) => void;
  saveWorkspace: () => Promise<void>;
  exportWorkspace: (id: string) => string;
  importWorkspace: (data: string) => Promise<WorkspaceLayout>;
  createFolder: (name: string, description: string, parentId?: string) => Promise<WorkspaceFolder>;
  updateFolder: (id: string, updates: Partial<WorkspaceFolder>) => Promise<void>;
  deleteFolder: (id: string) => Promise<void>;
  moveWorkspace: (workspaceId: string, folderId: string) => Promise<void>;
}

const WorkspaceContext = createContext<WorkspaceContextType | null>(null);

export const useWorkspace = () => {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error('useWorkspace must be used within a WorkspaceProvider');
  }
  return context;
};

// Workspace Provider
export const WorkspaceProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [workspaces, setWorkspaces] = useState<WorkspaceLayout[]>([]);
  const [folders, setFolders] = useState<WorkspaceFolder[]>([]);
  const [activeWorkspace, setActiveWorkspaceState] = useState<WorkspaceLayout | null>(null);
  const [templates, setTemplates] = useState<WorkspaceTemplate[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load data on mount
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      const mockWorkspaces: WorkspaceLayout[] = [
        {
          id: 'workspace-1',
          name: 'Trading Dashboard',
          description: 'Main trading workspace with charts and order management',
          category: 'trading',
          isTemplate: false,
          isPublic: false,
          isFavorite: true,
          tags: ['trading', 'charts', 'orders'],
          layout: {
            type: 'grid',
            gridSize: { columns: 12, rows: 8 },
            snapToGrid: true,
            gridSpacing: 8,
            backgroundColor: '#ffffff',
            widgets: []
          },
          permissions: {
            owner: 'user-1',
            editors: [],
            viewers: [],
            isPublic: false
          },
          metadata: {
            createdBy: 'user-1',
            createdAt: Date.now() - 86400000,
            updatedBy: 'user-1',
            updatedAt: Date.now() - 3600000,
            version: 1,
            usageCount: 25,
            rating: 4.5,
            reviews: []
          },
          settings: {
            autoSave: true,
            autoSaveInterval: 30000,
            showGrid: true,
            showRuler: false,
            enableCollaboration: false,
            allowComments: false,
            trackChanges: true
          }
        }
      ];

      const mockFolders: WorkspaceFolder[] = [
        {
          id: 'folder-1',
          name: 'Trading Workspaces',
          description: 'Workspaces for active trading',
          color: '#3b82f6',
          icon: 'TrendingUp',
          workspaces: ['workspace-1'],
          subfolders: [],
          permissions: {
            owner: 'user-1',
            editors: [],
            viewers: []
          },
          createdAt: Date.now() - 86400000 * 7,
          updatedAt: Date.now() - 86400000
        }
      ];

      const mockTemplates: WorkspaceTemplate[] = [
        {
          id: 'template-1',
          name: 'Day Trading Setup',
          description: 'Optimized layout for day trading with multiple timeframes',
          category: 'trading',
          preview: '/templates/day-trading.png',
          layout: mockWorkspaces[0],
          popularity: 95,
          downloads: 1250
        }
      ];

      setWorkspaces(mockWorkspaces);
      setFolders(mockFolders);
      setTemplates(mockTemplates);
      setActiveWorkspaceState(mockWorkspaces[0]);
      setIsLoading(false);
    };

    loadData();
  }, []);

  const createWorkspace = useCallback(async (name: string, description: string, category: string): Promise<WorkspaceLayout> => {
    const newWorkspace: WorkspaceLayout = {
      id: `workspace-${Date.now()}`,
      name,
      description,
      category: category as any,
      isTemplate: false,
      isPublic: false,
      isFavorite: false,
      tags: [],
      layout: {
        type: 'grid',
        gridSize: { columns: 12, rows: 8 },
        snapToGrid: true,
        gridSpacing: 8,
        backgroundColor: '#ffffff',
        widgets: []
      },
      permissions: {
        owner: 'current-user',
        editors: [],
        viewers: [],
        isPublic: false
      },
      metadata: {
        createdBy: 'current-user',
        createdAt: Date.now(),
        updatedBy: 'current-user',
        updatedAt: Date.now(),
        version: 1,
        usageCount: 0,
        rating: 0,
        reviews: []
      },
      settings: {
        autoSave: true,
        autoSaveInterval: 30000,
        showGrid: true,
        showRuler: false,
        enableCollaboration: false,
        allowComments: false,
        trackChanges: true
      }
    };

    setWorkspaces(prev => [...prev, newWorkspace]);
    return newWorkspace;
  }, []);

  const updateWorkspace = useCallback(async (id: string, updates: Partial<WorkspaceLayout>) => {
    setWorkspaces(prev => prev.map(workspace => 
      workspace.id === id 
        ? { 
            ...workspace, 
            ...updates, 
            metadata: { 
              ...workspace.metadata, 
              updatedAt: Date.now(),
              version: workspace.metadata.version + 1
            }
          }
        : workspace
    ));

    if (activeWorkspace?.id === id) {
      setActiveWorkspaceState(prev => prev ? { ...prev, ...updates } : null);
    }
  }, [activeWorkspace]);

  const deleteWorkspace = useCallback(async (id: string) => {
    setWorkspaces(prev => prev.filter(workspace => workspace.id !== id));
    if (activeWorkspace?.id === id) {
      setActiveWorkspaceState(null);
    }
  }, [activeWorkspace]);

  const duplicateWorkspace = useCallback(async (id: string, newName: string): Promise<WorkspaceLayout> => {
    const original = workspaces.find(w => w.id === id);
    if (!original) throw new Error('Workspace not found');

    const duplicate: WorkspaceLayout = {
      ...original,
      id: `workspace-${Date.now()}`,
      name: newName,
      metadata: {
        ...original.metadata,
        createdBy: 'current-user',
        createdAt: Date.now(),
        updatedBy: 'current-user',
        updatedAt: Date.now(),
        version: 1,
        usageCount: 0
      }
    };

    setWorkspaces(prev => [...prev, duplicate]);
    return duplicate;
  }, [workspaces]);

  const setActiveWorkspace = useCallback((workspace: WorkspaceLayout | null) => {
    setActiveWorkspaceState(workspace);
  }, []);

  const addWidget = useCallback((widget: Omit<WorkspaceWidget, 'id' | 'createdAt' | 'updatedAt'>) => {
    if (!activeWorkspace) return;

    const newWidget: WorkspaceWidget = {
      ...widget,
      id: `widget-${Date.now()}`,
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    const updatedWorkspace = {
      ...activeWorkspace,
      layout: {
        ...activeWorkspace.layout,
        widgets: [...activeWorkspace.layout.widgets, newWidget]
      }
    };

    updateWorkspace(activeWorkspace.id, updatedWorkspace);
  }, [activeWorkspace, updateWorkspace]);

  const updateWidget = useCallback((widgetId: string, updates: Partial<WorkspaceWidget>) => {
    if (!activeWorkspace) return;

    const updatedWidgets = activeWorkspace.layout.widgets.map(widget =>
      widget.id === widgetId 
        ? { ...widget, ...updates, updatedAt: Date.now() }
        : widget
    );

    updateWorkspace(activeWorkspace.id, {
      layout: { ...activeWorkspace.layout, widgets: updatedWidgets }
    });
  }, [activeWorkspace, updateWorkspace]);

  const removeWidget = useCallback((widgetId: string) => {
    if (!activeWorkspace) return;

    const updatedWidgets = activeWorkspace.layout.widgets.filter(widget => widget.id !== widgetId);

    updateWorkspace(activeWorkspace.id, {
      layout: { ...activeWorkspace.layout, widgets: updatedWidgets }
    });
  }, [activeWorkspace, updateWorkspace]);

  const saveWorkspace = useCallback(async () => {
    if (!activeWorkspace) return;
    
    // Simulate API save
    await new Promise(resolve => setTimeout(resolve, 500));
    
    updateWorkspace(activeWorkspace.id, {
      metadata: {
        ...activeWorkspace.metadata,
        updatedAt: Date.now()
      }
    });
  }, [activeWorkspace, updateWorkspace]);

  const exportWorkspace = useCallback((id: string): string => {
    const workspace = workspaces.find(w => w.id === id);
    if (!workspace) throw new Error('Workspace not found');
    
    return JSON.stringify(workspace, null, 2);
  }, [workspaces]);

  const importWorkspace = useCallback(async (data: string): Promise<WorkspaceLayout> => {
    try {
      const imported = JSON.parse(data);
      const newWorkspace: WorkspaceLayout = {
        ...imported,
        id: `workspace-${Date.now()}`,
        metadata: {
          ...imported.metadata,
          createdBy: 'current-user',
          createdAt: Date.now(),
          updatedBy: 'current-user',
          updatedAt: Date.now(),
          version: 1
        }
      };

      setWorkspaces(prev => [...prev, newWorkspace]);
      return newWorkspace;
    } catch (error) {
      throw new Error('Invalid workspace data');
    }
  }, []);

  const createFolder = useCallback(async (name: string, description: string, parentId?: string): Promise<WorkspaceFolder> => {
    const newFolder: WorkspaceFolder = {
      id: `folder-${Date.now()}`,
      name,
      description,
      color: '#3b82f6',
      icon: 'Folder',
      parentId,
      workspaces: [],
      subfolders: [],
      permissions: {
        owner: 'current-user',
        editors: [],
        viewers: []
      },
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    setFolders(prev => [...prev, newFolder]);
    return newFolder;
  }, []);

  const updateFolder = useCallback(async (id: string, updates: Partial<WorkspaceFolder>) => {
    setFolders(prev => prev.map(folder =>
      folder.id === id 
        ? { ...folder, ...updates, updatedAt: Date.now() }
        : folder
    ));
  }, []);

  const deleteFolder = useCallback(async (id: string) => {
    setFolders(prev => prev.filter(folder => folder.id !== id));
  }, []);

  const moveWorkspace = useCallback(async (workspaceId: string, folderId: string) => {
    // Remove workspace from all folders
    setFolders(prev => prev.map(folder => ({
      ...folder,
      workspaces: folder.workspaces.filter(id => id !== workspaceId)
    })));

    // Add workspace to target folder
    setFolders(prev => prev.map(folder =>
      folder.id === folderId
        ? { ...folder, workspaces: [...folder.workspaces, workspaceId] }
        : folder
    ));
  }, []);

  const contextValue: WorkspaceContextType = {
    workspaces,
    folders,
    activeWorkspace,
    templates,
    isLoading,
    createWorkspace,
    updateWorkspace,
    deleteWorkspace,
    duplicateWorkspace,
    setActiveWorkspace,
    addWidget,
    updateWidget,
    removeWidget,
    saveWorkspace,
    exportWorkspace,
    importWorkspace,
    createFolder,
    updateFolder,
    deleteFolder,
    moveWorkspace
  };

  return (
    <WorkspaceContext.Provider value={contextValue}>
      {children}
    </WorkspaceContext.Provider>
  );
};

// Workspace Manager Component
export const WorkspaceManager: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const {
    workspaces,
    folders,
    activeWorkspace,
    templates,
    isLoading,
    createWorkspace,
    updateWorkspace,
    deleteWorkspace,
    duplicateWorkspace,
    setActiveWorkspace,
    exportWorkspace,
    importWorkspace,
    createFolder,
    moveWorkspace
  } = useWorkspace();

  const [activeTab, setActiveTab] = useState<'workspaces' | 'templates' | 'folders'>('workspaces');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'updated' | 'created' | 'usage'>('updated');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);
  const [selectedWorkspace, setSelectedWorkspace] = useState<WorkspaceLayout | null>(null);

  const filteredWorkspaces = useMemo(() => {
    let filtered = workspaces.filter(workspace => {
      const matchesSearch = workspace.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           workspace.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           workspace.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      const matchesCategory = filterCategory === 'all' || workspace.category === filterCategory;
      return matchesSearch && matchesCategory;
    });

    // Sort workspaces
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'updated':
          aValue = a.metadata.updatedAt;
          bValue = b.metadata.updatedAt;
          break;
        case 'created':
          aValue = a.metadata.createdAt;
          bValue = b.metadata.createdAt;
          break;
        case 'usage':
          aValue = a.metadata.usageCount;
          bValue = b.metadata.usageCount;
          break;
        default:
          return 0;
      }

      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });

    return filtered;
  }, [workspaces, searchTerm, filterCategory, sortBy, sortOrder]);

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-6xl h-[85vh] mx-4 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="border-b border-gray-200 dark:border-gray-700 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Layout className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Workspace Manager
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Workspace
            </button>
            
            <button
              onClick={() => setShowImportModal(true)}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <Upload className="w-4 h-4" />
            </button>
            
            <button
              onClick={onClose}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700 px-6">
          <div className="flex gap-1">
            {([
              { id: 'workspaces', label: 'My Workspaces', icon: Layout },
              { id: 'templates', label: 'Templates', icon: Copy },
              { id: 'folders', label: 'Folders', icon: Folder }
            ] as const).map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search workspaces..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>

            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="all">All Categories</option>
              <option value="trading">Trading</option>
              <option value="analysis">Analysis</option>
              <option value="portfolio">Portfolio</option>
              <option value="research">Research</option>
              <option value="custom">Custom</option>
            </select>

            <select
              value={`${sortBy}-${sortOrder}`}
              onChange={(e) => {
                const [field, order] = e.target.value.split('-');
                setSortBy(field as any);
                setSortOrder(order as any);
              }}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="updated-desc">Recently Updated</option>
              <option value="created-desc">Recently Created</option>
              <option value="name-asc">Name A-Z</option>
              <option value="name-desc">Name Z-A</option>
              <option value="usage-desc">Most Used</option>
            </select>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          <AnimatePresence mode="wait">
            {activeTab === 'workspaces' && (
              <WorkspacesPanel
                key="workspaces"
                workspaces={filteredWorkspaces}
                activeWorkspace={activeWorkspace}
                onSelectWorkspace={setActiveWorkspace}
                onUpdateWorkspace={updateWorkspace}
                onDeleteWorkspace={deleteWorkspace}
                onDuplicateWorkspace={duplicateWorkspace}
                onExportWorkspace={exportWorkspace}
                isLoading={isLoading}
              />
            )}
            {activeTab === 'templates' && (
              <TemplatesPanel
                key="templates"
                templates={templates}
                onCreateFromTemplate={(template) => {
                  // Create workspace from template
                  createWorkspace(
                    `${template.name} Copy`,
                    template.description,
                    template.category
                  );
                }}
                isLoading={isLoading}
              />
            )}
            {activeTab === 'folders' && (
              <FoldersPanel
                key="folders"
                folders={folders}
                workspaces={workspaces}
                onCreateFolder={createFolder}
                onMoveWorkspace={moveWorkspace}
                isLoading={isLoading}
              />
            )}
          </AnimatePresence>
        </div>

        {/* Create Workspace Modal */}
        <AnimatePresence>
          {showCreateModal && (
            <CreateWorkspaceModal
              onClose={() => setShowCreateModal(false)}
              onCreate={async (name, description, category) => {
                const workspace = await createWorkspace(name, description, category);
                setActiveWorkspace(workspace);
                setShowCreateModal(false);
                onClose();
              }}
            />
          )}
        </AnimatePresence>

        {/* Import Workspace Modal */}
        <AnimatePresence>
          {showImportModal && (
            <ImportWorkspaceModal
              onClose={() => setShowImportModal(false)}
              onImport={async (data) => {
                const workspace = await importWorkspace(data);
                setActiveWorkspace(workspace);
                setShowImportModal(false);
              }}
            />
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
};

// Individual Panel Components (simplified for brevity)
const WorkspacesPanel: React.FC<any> = ({ workspaces, activeWorkspace, onSelectWorkspace, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="p-6 overflow-y-auto"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {workspaces.map((workspace) => (
          <motion.div
            key={workspace.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`p-4 border-2 rounded-lg cursor-pointer transition-all hover:shadow-lg ${
              activeWorkspace?.id === workspace.id
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
            onClick={() => onSelectWorkspace(workspace)}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <Layout className="w-5 h-5 text-blue-500" />
                {workspace.isFavorite && <Star className="w-4 h-4 text-yellow-500 fill-current" />}
              </div>
              <span className={`px-2 py-1 text-xs rounded-full ${
                workspace.category === 'trading' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
                workspace.category === 'analysis' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400' :
                workspace.category === 'portfolio' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400' :
                'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
              }`}>
                {workspace.category}
              </span>
            </div>
            
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{workspace.name}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 line-clamp-2">{workspace.description}</p>
            
            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>{workspace.layout.widgets.length} widgets</span>
              <span>{new Date(workspace.metadata.updatedAt).toLocaleDateString()}</span>
            </div>
          </motion.div>
        ))}
      </div>

      {workspaces.length === 0 && (
        <div className="text-center py-12">
          <Layout className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No workspaces found</h3>
          <p className="text-gray-600 dark:text-gray-400">Create your first workspace to get started</p>
        </div>
      )}
    </motion.div>
  );
};

// Additional panel components would be implemented similarly...
const TemplatesPanel: React.FC<any> = () => <div>Templates Panel</div>;
const FoldersPanel: React.FC<any> = () => <div>Folders Panel</div>;
const CreateWorkspaceModal: React.FC<any> = () => <div>Create Workspace Modal</div>;
const ImportWorkspaceModal: React.FC<any> = () => <div>Import Workspace Modal</div>;

// Export all components
export {
  WorkspacesPanel,
  TemplatesPanel,
  FoldersPanel,
  CreateWorkspaceModal,
  ImportWorkspaceModal,
  WIDGET_TYPES
};

