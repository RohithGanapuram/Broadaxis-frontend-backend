import React from 'react';
 
interface StatItem {
  title: string;
  count: number;
  subtitle: string;
  color: string;
  icon: string;
}
 
const StatsCards: React.FC = () => {
  const stats: StatItem[] = [
    {
      title: 'RFPs Today',
      count: 16,
      subtitle: 'Received today',
      color: '#3b82f6',
      icon: '📄'
    },
    {
      title: 'RFIs Today',
      count: 8,
      subtitle: 'Received today',
      color: '#8b5cf6',
      icon: '✉️'
    },
    {
      title: 'RFQs Today',
      count: 6,
      subtitle: 'Received today',
      color: '#f59e0b',
      icon: '📋'
    },
    {
      title: 'Go Recommendations',
      count: 8,
      subtitle: 'Approved',
      color: '#10b981',
      icon: '✅'
    },
    {
      title: 'No-Go Recommendations',
      count: 4,
      subtitle: 'Declined',
      color: '#ef4444',
      icon: '❌'
    }
  ];
 
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
      {stats.map((stat, index) => (
        <div key={index} className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 hover:shadow-xl transition-shadow duration-200">
          <div className="flex items-center justify-between mb-4">
            <span className="text-2xl">{stat.icon}</span>
            <h3 className="text-sm font-medium text-gray-600 text-right">{stat.title}</h3>
          </div>
          <div className="text-3xl font-bold mb-2" style={{ color: stat.color }}>
            {stat.count}
          </div>
          <div className="text-sm text-gray-500">{stat.subtitle}</div>
        </div>
      ))}
    </div>
  );
};
 
export default StatsCards;
 