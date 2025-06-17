#!/usr/bin/env python3
"""
WS3-P1 Step 2: Database Schema Implementation
ALL-USE Account Management System - Database Layer
"""

import sqlite3
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import os
from dataclasses import asdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from account_models import (
    BaseAccount, GenerationAccount, RevenueAccount, CompoundingAccount,
    AccountType, AccountStatus, TransactionType, Transaction, 
    PerformanceMetrics, AccountConfiguration, create_account
)

class AccountDatabase:
    """Database manager for ALL-USE account management system."""
    
    def __init__(self, db_path: str = "data/alluse_accounts.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create persistent connection and cursor for forking protocol compatibility
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        
        # Initialize database schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create database tables and indexes."""
        cursor = self.cursor
        
        # Create accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                account_name TEXT NOT NULL,
                account_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                initial_balance REAL NOT NULL,
                current_balance REAL NOT NULL,
                available_balance REAL NOT NULL,
                cash_buffer REAL NOT NULL,
                parent_account_id TEXT,
                forked_from_account_id TEXT,
                configuration TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                last_activity_at TIMESTAMP NOT NULL,
                FOREIGN KEY (parent_account_id) REFERENCES accounts (account_id),
                FOREIGN KEY (forked_from_account_id) REFERENCES accounts (account_id)
            )
        """)
        
        # Create account relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_account_id TEXT NOT NULL,
                child_account_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (parent_account_id) REFERENCES accounts (account_id),
                FOREIGN KEY (child_account_id) REFERENCES accounts (account_id),
                UNIQUE(parent_account_id, child_account_id)
            )
        """)
        
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                related_account_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT,
                FOREIGN KEY (account_id) REFERENCES accounts (account_id)
            )
        """)
        
        self.connection.commit()
