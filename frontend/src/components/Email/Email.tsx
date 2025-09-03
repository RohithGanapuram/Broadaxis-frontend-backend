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
  // Group attachments by email subject so multiple files from the same email appear together
  const groupedAttachments = React.useMemo(() => {
    const groups: Record<string, { subject: string; date?: string; items: EmailAttachment[] }> = {};
    for (const att of emailAttachments) {
      const subj = att.email_subject || "No Subject";
      if (!groups[subj]) {
        groups[subj] = { subject: subj, date: att.email_date, items: [] };
      }
      groups[subj].items.push(att);
    }
    // newest first
    return Object.values(groups).sort((a, b) => {
      const ta = a.date ? Date.parse(a.date) : 0;
      const tb = b.date ? Date.parse(b.date) : 0;
      return tb - ta;
    });
  }, [emailAttachments]);

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
                  {groupedAttachments.length > 0 ? (
                    <div className="space-y-4">
                      {groupedAttachments.map((group, gi) => (
                        <div key={gi} className="border border-gray-200 rounded-lg">
                          {/* Subject header */}
                          <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
                            <div className="font-semibold text-blue-700 truncate">
                              📧 {group.subject}
                            </div>
                            {group.date && (
                              <div className="text-xs text-gray-500">
                                📅 {new Date(group.date).toLocaleDateString()}
                              </div>
                            )}
                          </div>
                          {/* Attachments under this subject */}
                          <div className="p-3 space-y-2">
                            {group.items.map((att, i) => (
                              <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded-md">
                                <div className="flex items-center space-x-3">
                                  <div className="text-lg">
                                    {att.type === 'link' ? '🔗' :
                                    att.filename?.endsWith('.pdf') ? '📄' :
                                    att.filename?.endsWith('.docx') ? '📝' :
                                    att.filename?.endsWith('.xlsx') ? '📊' : '📁'}
                                  </div>
                                  <div>
                                    <div className="text-xs font-medium text-gray-700">
                                      {att.filename}
                                    </div>
                                    <div className="text-xs text-gray-500">
                                      {att.type === 'link'
                                        ? `Link • ${att.domain || 'External'}`
                                        : `${att.date || ''} ${att.size ? '• ' + att.size : ''}`}
                                    </div>
                                  </div>
                                </div>
                                {att.type === 'link' ? (
                                  <a
                                    href={att.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs bg-blue-600 text-white px-2 py-1 rounded"
                                  >
                                    Open
                                  </a>
                                ) : (
                                  <button className="text-xs text-blue-600">View</button>
                                )}
                              </div>
                            ))}
                          </div>
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