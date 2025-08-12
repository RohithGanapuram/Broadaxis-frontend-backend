import React, { useState, useEffect } from 'react';
import { apiClient } from '../../utils/api';

interface EmailAccount {
  id: number;
  email: string;
  count: number;
  label: string;
  subject: string;
  bgColor: string;
  textColor: string;
  iconColor: string;
  latest_date?: string;
  latest_file?: string;
}

interface EmailAttachment {
  filename: string;
  date?: string;
  size?: string;
  type?: 'file' | 'link';
  url?: string;
  domain?: string;
  email_subject?: string;
  email_date?: string;
}

interface FetchedEmailData {
  emails: EmailAccount[];
  total_count: number;
  total_files: number;
}
 
interface User {
  name: string;
}
 
const Email: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [fetchStatus, setFetchStatus] = useState('');
  const [emailData, setEmailData] = useState<FetchedEmailData>({
    emails: [],
    total_count: 0,
    total_files: 0
  });
  const [selectedEmail, setSelectedEmail] = useState<number | null>(null);
  const [emailAttachments, setEmailAttachments] = useState<EmailAttachment[]>([]);
  const [testStatus, setTestStatus] = useState('');


  const getCurrentUser = (): User => {
    const user = localStorage.getItem('user');
    if (user) {
      return JSON.parse(user);
    }
    return { name: 'User' };
  };

  const currentUser = getCurrentUser();

  // Load fetched emails on component mount
  useEffect(() => {
    loadFetchedEmails();
  }, []);

  const loadFetchedEmails = async () => {
    try {
      const data = await apiClient.getFetchedEmails();
      setEmailData(data);
    } catch (error) {
      console.error('Failed to load fetched emails:', error);
    }
  };

  const handleTestAuth = async () => {
    setTestStatus('🧪 Testing Microsoft Graph API authentication...');
    
    try {
      const result = await apiClient.testGraphAuth();
      
      if (result.status === 'success') {
        setTestStatus('✅ Microsoft Graph API authentication working!');
      } else if (result.step === 'permissions') {
        const instructions = result.fix_instructions || [];
        const instructionText = instructions.join('\n');
        setTestStatus(`❌ ${result.message}\n\n🔧 How to fix:\n${instructionText}`);
      } else {
        setTestStatus(`❌ Auth test failed: ${result.message}`);
      }
    } catch (error) {
      setTestStatus(`❌ Auth test error: ${error}`);
    }
  };

  const handleFetchEmails = async () => {
    setIsLoading(true);
    setFetchStatus('🔍 Connecting to Microsoft Graph API and searching for RFP/RFI/RFQ emails...');

    try {
      // Always use real email with Graph API
      const result = await apiClient.fetchEmails([], true, true);

      if (result.status === 'success') {
        setFetchStatus(`✅ Successfully fetched ${result.emails_found} emails with ${result.attachments_downloaded} attachments!`);
        // Reload the email data
        await loadFetchedEmails();
      } else {
        setFetchStatus(`❌ Error: ${result.message}`);
      }
    } catch (error) {
      setFetchStatus(`❌ Failed to fetch emails: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEmailCardClick = async (emailId: number) => {
    if (selectedEmail === emailId) {
      setSelectedEmail(null);
      setEmailAttachments([]);
    } else {
      setSelectedEmail(emailId);
      try {
        const attachmentData = await apiClient.getEmailAttachments(emailId);
        setEmailAttachments(attachmentData.attachments || []);
      } catch (error) {
        console.error('Failed to load email attachments:', error);
        setEmailAttachments([]);
      }
    }
  };
 
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
 
  // Use dynamic email data or fallback to static data
  const emailAccounts: EmailAccount[] = emailData.emails.length > 0 ?
    emailData.emails.map(email => ({
      ...email,
      label: 'RFP/RFI/RFQ Emails',
      subject: email.latest_file || 'No recent files',
      bgColor: email.id === 1 ? '#f0fdf4' : email.id === 2 ? '#faf5ff' : '#fff7ed',
      textColor: email.id === 1 ? '#16a34a' : email.id === 2 ? '#9333ea' : '#ea580c',
      iconColor: email.id === 1 ? '#16a34a' : email.id === 2 ? '#9333ea' : '#ea580c'
    })) : [
    {
      id: 1,
      email: 'proposals@broadaxis.com',
      count: 0,
      label: 'RFP/RFI/RFQ Emails',
      subject: 'No emails fetched yet',
      bgColor: '#f0fdf4',
      textColor: '#16a34a',
      iconColor: '#16a34a'
    },
    {
      id: 2,
      email: 'rfp.team@broadaxis.com',
      count: 0,
      label: 'RFP/RFI/RFQ Emails',
      subject: 'No emails fetched yet',
      bgColor: '#faf5ff',
      textColor: '#9333ea',
      iconColor: '#9333ea'
    },
    {
      id: 3,
      email: 'business@broadaxis.com',
      count: 0,
      label: 'RFP/RFI/RFQ Emails',
      subject: 'No emails fetched yet',
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
        <div className="flex space-x-4 mb-4">
          <button
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-all duration-200 shadow-lg"
            onClick={handleTestAuth}
          >
            🧪 Test Auth
          </button>
          <button
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              isLoading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 shadow-lg'
            } text-white`}
            onClick={handleFetchEmails}
            disabled={isLoading}
          >
            {isLoading ? '🔄 Fetching...' : '📧 Fetch RFP/RFI/RFQ Emails'}
          </button>
        </div>



        {testStatus && (
          <div className={`mb-4 p-3 rounded-lg text-sm ${
            testStatus.includes('failed') || testStatus.includes('error')
              ? 'bg-red-50 text-red-700 border border-red-200'
              : 'bg-green-50 text-green-700 border border-green-200'
          }`}>
            {testStatus}
          </div>
        )}
        
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
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Total RFP/RFI/RFQ Emails</h2>
        <div className="text-4xl font-bold text-blue-600 mb-2">{emailData.total_count}</div>
        <div className="text-gray-600">
          {emailData.total_files > 0 ? `${emailData.total_files} attachments downloaded` : 'No emails fetched yet'}
        </div>
      </div>
 
      {/* Email Accounts Section */}
      <div className="space-y-4">
        <h3 className="text-xl font-bold text-gray-900">Email Accounts</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {emailAccounts.map((account) => (
            <div key={account.id} className="space-y-4">
              <div
                className={`rounded-xl shadow-lg border border-gray-100 p-6 cursor-pointer transition-all duration-200 hover:shadow-xl ${
                  selectedEmail === account.id ? 'ring-2 ring-blue-500' : ''
                }`}
                style={{ backgroundColor: account.bgColor }}
                onClick={() => handleEmailCardClick(account.id)}
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
                  <div className="text-xs font-medium text-gray-500">Latest File:</div>
                  <div className="text-sm text-gray-700">{account.subject}</div>
                  {account.latest_date && (
                    <div className="text-xs text-gray-500">
                      Last updated: {new Date(account.latest_date).toLocaleDateString()}
                    </div>
                  )}
                </div>

                <div className="mt-4 text-xs text-blue-600 font-medium">
                  {selectedEmail === account.id ? '▼ Hide attachments' : '▶ Click to view attachments'}
                </div>
              </div>

              {/* Attachments Panel */}
              {selectedEmail === account.id && (
                <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-4">
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                    📎 Attachments ({emailAttachments.length})
                  </h4>
                  {emailAttachments.length > 0 ? (
                    <div className="space-y-2">
                      {emailAttachments.map((attachment, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <div className="text-lg">
                              {attachment.type === 'link' ? '🔗' :
                               attachment.filename?.endsWith('.pdf') ? '📄' :
                               attachment.filename?.endsWith('.docx') ? '📝' :
                               attachment.filename?.endsWith('.xlsx') ? '📊' : '📁'}
                            </div>
                            <div>
                              {attachment.email_subject && (
                                <div className="text-sm text-blue-600 font-bold mb-1">
                                  📧 {attachment.email_subject}
                                </div>
                              )}
                              <div className="text-xs font-medium text-gray-700">
                                {attachment.filename}
                              </div>
                              {attachment.email_date && (
                                <div className="text-xs text-gray-400 mb-1">
                                  📅 {new Date(attachment.email_date).toLocaleDateString()}
                                </div>
                              )}
                              <div className="text-xs text-gray-500">
                                {attachment.type === 'link' ?
                                  `Link • ${attachment.domain || 'External'}` :
                                  `${attachment.date} • ${attachment.size}`
                                }
                              </div>
                            </div>
                          </div>
                          {attachment.type === 'link' ? (
                            <a
                              href={attachment.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 bg-blue-600 hover:bg-blue-700 text-white px-2 py-1 rounded text-xs font-medium transition-colors duration-200"
                            >
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                              Open
                            </a>
                          ) : (
                            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                              View
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-gray-500 text-sm text-center py-4">
                      No attachments found for this email account
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
 
export default Email;