// Advanced Gesture Recognition System for WS6-P3
// Multi-touch gestures, drag-and-drop, and sophisticated input handling

import { useCallback, useEffect, useRef, useState } from 'react';

// Gesture Types and Interfaces
export interface GestureEvent {
  type: 'swipe' | 'pinch' | 'rotate' | 'tap' | 'longPress' | 'drag';
  direction?: 'left' | 'right' | 'up' | 'down';
  distance?: number;
  velocity?: number;
  scale?: number;
  rotation?: number;
  center?: { x: number; y: number };
  startPoint?: { x: number; y: number };
  endPoint?: { x: number; y: number };
  duration?: number;
  touches?: number;
}

export interface GestureConfig {
  swipe?: {
    minDistance?: number;
    maxTime?: number;
    threshold?: number;
  };
  pinch?: {
    minScale?: number;
    maxScale?: number;
    threshold?: number;
  };
  rotate?: {
    threshold?: number;
  };
  tap?: {
    maxTime?: number;
    maxDistance?: number;
  };
  longPress?: {
    duration?: number;
    maxDistance?: number;
  };
  drag?: {
    threshold?: number;
  };
}

// Advanced Gesture Recognition Hook
export const useAdvancedGestures = (
  config: GestureConfig = {},
  onGesture?: (gesture: GestureEvent) => void
) => {
  const [isGesturing, setIsGesturing] = useState(false);
  const [currentGesture, setCurrentGesture] = useState<GestureEvent | null>(null);
  
  const touchStart = useRef<{ [key: number]: Touch }>({});
  const touchCurrent = useRef<{ [key: number]: Touch }>({});
  const gestureStart = useRef<{
    time: number;
    center: { x: number; y: number };
    distance: number;
    angle: number;
  } | null>(null);
  
  const longPressTimer = useRef<NodeJS.Timeout | null>(null);
  const element = useRef<HTMLElement | null>(null);

  // Default configuration
  const defaultConfig: Required<GestureConfig> = {
    swipe: {
      minDistance: 50,
      maxTime: 300,
      threshold: 10
    },
    pinch: {
      minScale: 0.5,
      maxScale: 3,
      threshold: 0.1
    },
    rotate: {
      threshold: 15
    },
    tap: {
      maxTime: 200,
      maxDistance: 10
    },
    longPress: {
      duration: 500,
      maxDistance: 10
    },
    drag: {
      threshold: 5
    }
  };

  const mergedConfig = { ...defaultConfig, ...config };

  // Utility functions
  const getDistance = useCallback((touch1: Touch, touch2: Touch): number => {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }, []);

  const getAngle = useCallback((touch1: Touch, touch2: Touch): number => {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.atan2(dy, dx) * 180 / Math.PI;
  }, []);

  const getCenter = useCallback((touches: Touch[]): { x: number; y: number } => {
    const x = touches.reduce((sum, touch) => sum + touch.clientX, 0) / touches.length;
    const y = touches.reduce((sum, touch) => sum + touch.clientY, 0) / touches.length;
    return { x, y };
  }, []);

  const getTouchArray = useCallback((touchList: TouchList): Touch[] => {
    return Array.from(touchList);
  }, []);

  // Touch event handlers
  const handleTouchStart = useCallback((event: TouchEvent) => {
    const touches = getTouchArray(event.touches);
    const time = Date.now();

    // Store touch start positions
    touches.forEach(touch => {
      touchStart.current[touch.identifier] = touch;
      touchCurrent.current[touch.identifier] = touch;
    });

    if (touches.length === 1) {
      // Single touch - potential tap, long press, or drag
      const touch = touches[0];
      
      // Start long press timer
      longPressTimer.current = setTimeout(() => {
        const currentTouch = touchCurrent.current[touch.identifier];
        if (currentTouch) {
          const distance = getDistance(touch, currentTouch);
          if (distance <= mergedConfig.longPress.maxDistance) {
            const gesture: GestureEvent = {
              type: 'longPress',
              center: { x: touch.clientX, y: touch.clientY },
              duration: Date.now() - time,
              touches: 1
            };
            setCurrentGesture(gesture);
            onGesture?.(gesture);
          }
        }
      }, mergedConfig.longPress.duration);

    } else if (touches.length === 2) {
      // Two touches - potential pinch or rotate
      const [touch1, touch2] = touches;
      const distance = getDistance(touch1, touch2);
      const angle = getAngle(touch1, touch2);
      const center = getCenter(touches);

      gestureStart.current = {
        time,
        center,
        distance,
        angle
      };
    }

    setIsGesturing(true);
  }, [mergedConfig, onGesture, getTouchArray, getDistance, getAngle, getCenter]);

  const handleTouchMove = useCallback((event: TouchEvent) => {
    event.preventDefault();
    const touches = getTouchArray(event.touches);

    // Update current touch positions
    touches.forEach(touch => {
      touchCurrent.current[touch.identifier] = touch;
    });

    if (touches.length === 1) {
      // Single touch movement - potential drag or swipe
      const touch = touches[0];
      const startTouch = touchStart.current[touch.identifier];
      
      if (startTouch) {
        const distance = getDistance(startTouch, touch);
        
        if (distance > mergedConfig.drag.threshold) {
          // Clear long press timer
          if (longPressTimer.current) {
            clearTimeout(longPressTimer.current);
            longPressTimer.current = null;
          }

          const gesture: GestureEvent = {
            type: 'drag',
            startPoint: { x: startTouch.clientX, y: startTouch.clientY },
            endPoint: { x: touch.clientX, y: touch.clientY },
            distance,
            touches: 1
          };
          setCurrentGesture(gesture);
          onGesture?.(gesture);
        }
      }

    } else if (touches.length === 2 && gestureStart.current) {
      // Two touch movement - pinch or rotate
      const [touch1, touch2] = touches;
      const currentDistance = getDistance(touch1, touch2);
      const currentAngle = getAngle(touch1, touch2);
      const currentCenter = getCenter(touches);

      const scale = currentDistance / gestureStart.current.distance;
      const rotation = currentAngle - gestureStart.current.angle;

      // Detect pinch
      if (Math.abs(scale - 1) > mergedConfig.pinch.threshold) {
        const gesture: GestureEvent = {
          type: 'pinch',
          scale,
          center: currentCenter,
          touches: 2
        };
        setCurrentGesture(gesture);
        onGesture?.(gesture);
      }

      // Detect rotation
      if (Math.abs(rotation) > mergedConfig.rotate.threshold) {
        const gesture: GestureEvent = {
          type: 'rotate',
          rotation,
          center: currentCenter,
          touches: 2
        };
        setCurrentGesture(gesture);
        onGesture?.(gesture);
      }
    }
  }, [mergedConfig, onGesture, getTouchArray, getDistance, getAngle, getCenter]);

  const handleTouchEnd = useCallback((event: TouchEvent) => {
    const touches = getTouchArray(event.changedTouches);
    const time = Date.now();

    touches.forEach(touch => {
      const startTouch = touchStart.current[touch.identifier];
      
      if (startTouch) {
        const distance = getDistance(startTouch, touch);
        const duration = time - (gestureStart.current?.time || time);

        // Clear long press timer
        if (longPressTimer.current) {
          clearTimeout(longPressTimer.current);
          longPressTimer.current = null;
        }

        // Detect tap
        if (duration <= mergedConfig.tap.maxTime && distance <= mergedConfig.tap.maxDistance) {
          const gesture: GestureEvent = {
            type: 'tap',
            center: { x: touch.clientX, y: touch.clientY },
            duration,
            touches: 1
          };
          setCurrentGesture(gesture);
          onGesture?.(gesture);
        }
        // Detect swipe
        else if (distance >= mergedConfig.swipe.minDistance && duration <= mergedConfig.swipe.maxTime) {
          const deltaX = touch.clientX - startTouch.clientX;
          const deltaY = touch.clientY - startTouch.clientY;
          const velocity = distance / duration;

          let direction: 'left' | 'right' | 'up' | 'down';
          if (Math.abs(deltaX) > Math.abs(deltaY)) {
            direction = deltaX > 0 ? 'right' : 'left';
          } else {
            direction = deltaY > 0 ? 'down' : 'up';
          }

          const gesture: GestureEvent = {
            type: 'swipe',
            direction,
            distance,
            velocity,
            startPoint: { x: startTouch.clientX, y: startTouch.clientY },
            endPoint: { x: touch.clientX, y: touch.clientY },
            duration,
            touches: 1
          };
          setCurrentGesture(gesture);
          onGesture?.(gesture);
        }

        // Clean up touch data
        delete touchStart.current[touch.identifier];
        delete touchCurrent.current[touch.identifier];
      }
    });

    // Reset gesture state if no more touches
    if (event.touches.length === 0) {
      setIsGesturing(false);
      setCurrentGesture(null);
      gestureStart.current = null;
    }
  }, [mergedConfig, onGesture, getTouchArray, getDistance]);

  // Attach event listeners
  const attachListeners = useCallback((el: HTMLElement) => {
    element.current = el;
    el.addEventListener('touchstart', handleTouchStart, { passive: false });
    el.addEventListener('touchmove', handleTouchMove, { passive: false });
    el.addEventListener('touchend', handleTouchEnd, { passive: false });
  }, [handleTouchStart, handleTouchMove, handleTouchEnd]);

  const detachListeners = useCallback(() => {
    if (element.current) {
      element.current.removeEventListener('touchstart', handleTouchStart);
      element.current.removeEventListener('touchmove', handleTouchMove);
      element.current.removeEventListener('touchend', handleTouchEnd);
      element.current = null;
    }
  }, [handleTouchStart, handleTouchMove, handleTouchEnd]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      detachListeners();
      if (longPressTimer.current) {
        clearTimeout(longPressTimer.current);
      }
    };
  }, [detachListeners]);

  return {
    isGesturing,
    currentGesture,
    attachListeners,
    detachListeners
  };
};

// Drag and Drop System
export interface DragDropConfig {
  dragThreshold?: number;
  snapToGrid?: boolean;
  gridSize?: number;
  constrainToParent?: boolean;
  axis?: 'x' | 'y' | 'both';
}

export const useDragDrop = (
  config: DragDropConfig = {},
  onDragStart?: (event: { x: number; y: number }) => void,
  onDrag?: (event: { x: number; y: number; deltaX: number; deltaY: number }) => void,
  onDragEnd?: (event: { x: number; y: number }) => void
) => {
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const element = useRef<HTMLElement | null>(null);

  const defaultConfig: Required<DragDropConfig> = {
    dragThreshold: 5,
    snapToGrid: false,
    gridSize: 20,
    constrainToParent: false,
    axis: 'both'
  };

  const mergedConfig = { ...defaultConfig, ...config };

  const handleMouseDown = useCallback((event: MouseEvent) => {
    dragStart.current = { x: event.clientX, y: event.clientY };
    setDragOffset({ x: 0, y: 0 });
    
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragStart.current) return;

      const deltaX = moveEvent.clientX - dragStart.current.x;
      const deltaY = moveEvent.clientY - dragStart.current.y;
      const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

      if (!isDragging && distance > mergedConfig.dragThreshold) {
        setIsDragging(true);
        onDragStart?.({ x: moveEvent.clientX, y: moveEvent.clientY });
      }

      if (isDragging) {
        let newX = deltaX;
        let newY = deltaY;

        // Apply axis constraints
        if (mergedConfig.axis === 'x') newY = 0;
        if (mergedConfig.axis === 'y') newX = 0;

        // Apply grid snapping
        if (mergedConfig.snapToGrid) {
          newX = Math.round(newX / mergedConfig.gridSize) * mergedConfig.gridSize;
          newY = Math.round(newY / mergedConfig.gridSize) * mergedConfig.gridSize;
        }

        // Apply parent constraints
        if (mergedConfig.constrainToParent && element.current?.parentElement) {
          const parent = element.current.parentElement;
          const parentRect = parent.getBoundingClientRect();
          const elementRect = element.current.getBoundingClientRect();

          const minX = parentRect.left - elementRect.left;
          const maxX = parentRect.right - elementRect.right;
          const minY = parentRect.top - elementRect.top;
          const maxY = parentRect.bottom - elementRect.bottom;

          newX = Math.max(minX, Math.min(maxX, newX));
          newY = Math.max(minY, Math.min(maxY, newY));
        }

        setDragOffset({ x: newX, y: newY });
        onDrag?.({ x: moveEvent.clientX, y: moveEvent.clientY, deltaX: newX, deltaY: newY });
      }
    };

    const handleMouseUp = (upEvent: MouseEvent) => {
      if (isDragging) {
        onDragEnd?.({ x: upEvent.clientX, y: upEvent.clientY });
      }
      
      setIsDragging(false);
      dragStart.current = null;
      
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [isDragging, mergedConfig, onDragStart, onDrag, onDragEnd]);

  const attachDragListeners = useCallback((el: HTMLElement) => {
    element.current = el;
    el.addEventListener('mousedown', handleMouseDown);
  }, [handleMouseDown]);

  const detachDragListeners = useCallback(() => {
    if (element.current) {
      element.current.removeEventListener('mousedown', handleMouseDown);
      element.current = null;
    }
  }, [handleMouseDown]);

  return {
    isDragging,
    dragOffset,
    attachDragListeners,
    detachDragListeners
  };
};

// Multi-touch Zoom and Pan
export const useZoomPan = (
  onZoom?: (scale: number, center: { x: number; y: number }) => void,
  onPan?: (offset: { x: number; y: number }) => void
) => {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  
  const lastTouchDistance = useRef<number | null>(null);
  const lastTouchCenter = useRef<{ x: number; y: number } | null>(null);
  const isPanning = useRef(false);

  const handleWheel = useCallback((event: WheelEvent) => {
    event.preventDefault();
    
    if (event.ctrlKey) {
      // Zoom
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(0.1, Math.min(5, scale * delta));
      setScale(newScale);
      onZoom?.(newScale, { x: event.clientX, y: event.clientY });
    } else {
      // Pan
      const newOffset = {
        x: offset.x - event.deltaX,
        y: offset.y - event.deltaY
      };
      setOffset(newOffset);
      onPan?.(newOffset);
    }
  }, [scale, offset, onZoom, onPan]);

  const { attachListeners: attachGestureListeners } = useAdvancedGestures(
    {
      pinch: { threshold: 0.05 },
      drag: { threshold: 3 }
    },
    (gesture) => {
      if (gesture.type === 'pinch' && gesture.scale) {
        const newScale = Math.max(0.1, Math.min(5, scale * gesture.scale));
        setScale(newScale);
        onZoom?.(newScale, gesture.center || { x: 0, y: 0 });
      } else if (gesture.type === 'drag' && gesture.touches === 1) {
        const deltaX = (gesture.endPoint?.x || 0) - (gesture.startPoint?.x || 0);
        const deltaY = (gesture.endPoint?.y || 0) - (gesture.startPoint?.y || 0);
        const newOffset = {
          x: offset.x + deltaX,
          y: offset.y + deltaY
        };
        setOffset(newOffset);
        onPan?.(newOffset);
      }
    }
  );

  const attachZoomPanListeners = useCallback((el: HTMLElement) => {
    el.addEventListener('wheel', handleWheel, { passive: false });
    attachGestureListeners(el);
  }, [handleWheel, attachGestureListeners]);

  const reset = useCallback(() => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  return {
    scale,
    offset,
    attachZoomPanListeners,
    reset
  };
};

// Export all gesture utilities
export {
  type GestureEvent,
  type GestureConfig,
  type DragDropConfig
};

