import React, { useState, useEffect } from 'react';
 

interface EmailAccount {
  id: number;
  email: string;
  count: number;
  label: string;
  subject: string;
  bgColor: string;
  textColor: string;
  iconColor: string;
}
 
interface User {
  name: string;
}
 
const Email: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [fetchStatus, setFetchStatus] = useState('');
 
  const getCurrentUser = (): User => {
    const user = localStorage.getItem('user');
    if (user) {
      return JSON.parse(user);
    }
    return { name: 'User' };
  };
 
  const currentUser = getCurrentUser();
 
  const handleFetchEmails = async () => {
    setIsLoading(true);
    setFetchStatus('Email fetching feature coming soon...');
    
    // Simulate loading for demo
    setTimeout(() => {
      setFetchStatus('Email integration will be available in the next update');
      setIsLoading(false);
    }, 2000);
  };
 
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
 
  const emailAccounts: EmailAccount[] = [
    {
      id: 1,
      email: 'proposals@co...',
      count: 26,
      label: 'Proposals Received',
      subject: 'RFP Response - Infrastructure Project',
      bgColor: '#f0fdf4',
      textColor: '#16a34a',
      iconColor: '#16a34a'
    },
    {
      id: 2,
      email: 'rfp.team@com...',
      count: 16,
      label: 'Proposals Received',
      subject: 'RFI Documentation - Software Solutions',
      bgColor: '#faf5ff',
      textColor: '#9333ea',
      iconColor: '#9333ea'
    },
    {
      id: 3,
      email: 'business@co...',
      count: 12,
      label: 'Proposals Received',
      subject: 'RFQ Inquiry - Hardware Procurement',
      bgColor: '#fff7ed',
      textColor: '#ea580c',
      iconColor: '#ea580c'
    }
  ];
 
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Email</h1>
          <div className="text-gray-600 mt-1">Welcome back, {currentUser.name}</div>
        </div>
      </div>
 
      {/* Email Fetch Section */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-2 flex items-center">
          <span className="mr-2">📥</span>
          Fetch RFP/RFI/RFQ Emails
        </h3>
        <p className="text-gray-600 mb-4">
          Automatically download PDFs from emails containing RFP, RFI, or RFQ keywords
        </p>
        <button
          className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
            isLoading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 shadow-lg'
          } text-white`}
          onClick={handleFetchEmails}
          disabled={isLoading}
        >
          {isLoading ? '🔄 Fetching...' : '📧 Fetch Emails'}
        </button>
        {fetchStatus && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${
            fetchStatus.includes('Error')
              ? 'bg-red-50 text-red-700 border border-red-200'
              : 'bg-green-50 text-green-700 border border-green-200'
          }`}>
            {fetchStatus}
          </div>
        )}
      </div>
 

      {/* Total Emails Card */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl shadow-lg border border-blue-200 p-6 text-center">
        <div className="text-4xl mb-4">📧</div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Total Emails Received</h2>
        <div className="text-4xl font-bold text-blue-600 mb-2">99</div>
        <div className="text-gray-600">Emails across all accounts</div>
      </div>
 
      {/* Email Accounts Section */}
      <div className="space-y-4">
        <h3 className="text-xl font-bold text-gray-900">Email Accounts</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {emailAccounts.map((account) => (
            <div
              key={account.id}
              className="rounded-xl shadow-lg border border-gray-100 p-6"
              style={{ backgroundColor: account.bgColor }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="text-2xl" style={{ color: account.iconColor }}>
                  ✉️
                </div>
                <div className="text-sm font-medium text-gray-700">{account.email}</div>
              </div>
 
              <div className="text-3xl font-bold mb-2" style={{ color: account.textColor }}>
                {account.count}
              </div>
 
              <div className="text-sm text-gray-600 mb-4">{account.label}</div>
 
              <div className="space-y-1">
                <div className="text-xs font-medium text-gray-500">Latest Subject:</div>
                <div className="text-sm text-gray-700">{account.subject}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
 
export default Email;