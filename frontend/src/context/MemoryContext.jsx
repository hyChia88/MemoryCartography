import React, { createContext, useState, useContext, useEffect } from 'react';
import { 
  getMemories, 
  searchMemories, 
  generateNarrative, 
  recordInteraction 
} from '../api/services';

// 创建 Context
const MemoryContext = createContext();

export const useMemory = () => useContext(MemoryContext);

export const MemoryProvider = ({ children }) => {
  // 状态管理
  const [userMemories, setUserMemories] = useState([]);
  const [publicMemories, setPublicMemories] = useState([]);
  const [activeDatabase, setActiveDatabase] = useState('user');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // 搜索和生成相关状态
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [narrative, setNarrative] = useState(null);
  
  // 初始加载记忆
  useEffect(() => {
    const loadInitialMemories = async () => {
      try {
        setIsLoading(true);
        const [userResult, publicResult] = await Promise.all([
          getMemories('user'),
          getMemories('public')
        ]);
        
        setUserMemories(userResult);
        setPublicMemories(publicResult);
        setError(null);
      } catch (err) {
        setError('Failed to load memories. Please try again later.');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadInitialMemories();
  }, []);
  
  // 切换数据库
  const toggleDatabase = () => {
    setActiveDatabase(activeDatabase === 'user' ? 'public' : 'user');
  };
  
  // 获取当前活跃数据库的记忆
  const currentMemories = activeDatabase === 'user' ? userMemories : publicMemories;
  
  // 搜索记忆
  const handleSearch = async (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      setNarrative(null);
      return;
    }
    
    try {
      setIsLoading(true);
      setSearchQuery(query);
      
      // 搜索记忆
      const results = await searchMemories(query, activeDatabase);
      setSearchResults(results);
      
      // 生成叙事
      const narrativeResult = await generateNarrative(query, activeDatabase);
      setNarrative(narrativeResult);
      
      setError(null);
    } catch (err) {
      setError('Search failed. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };
  
  // 处理记忆点击 - 记录交互并增加权重
  const handleMemoryClick = async (id) => {
    try {
      await recordInteraction(id, 'click');
      
      // 更新本地状态以反映交互
      const updateMemoryList = (memories) => {
        return memories.map(memory => {
          if (memory.id === id) {
            // 增加感知权重以获得即时反馈
            return { ...memory, weight: (memory.weight || 0) + 0.1 };
          }
          return memory;
        });
      };
      
      if (activeDatabase === 'user') {
        setUserMemories(updateMemoryList(userMemories));
      } else {
        setPublicMemories(updateMemoryList(publicMemories));
      }
      
      // 如果有搜索结果，也更新搜索结果
      if (searchResults.length > 0) {
        setSearchResults(updateMemoryList(searchResults));
      }
      
    } catch (err) {
      console.error('Failed to record interaction:', err);
    }
  };
  
  // Context 值
  const value = {
    userMemories,
    publicMemories,
    activeDatabase,
    toggleDatabase,
    isLoading,
    error,
    searchQuery,
    setSearchQuery,
    searchResults,
    narrative,
    handleSearch,
    handleMemoryClick,
    currentMemories
  };
  
  return (
    <MemoryContext.Provider value={value}>
      {children}
    </MemoryContext.Provider>
  );
};

export default MemoryContext;