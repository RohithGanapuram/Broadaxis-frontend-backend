import React from 'react';
import { apiClient } from '../../utils/api';

interface StatItem {
  title: string;
  count: number;
  subtitle: string;
  color: string;
  icon: string;
  isPreview?: boolean;
}

const isToday = (iso?: string) => {
  if (!iso) return false;
  const d = new Date(iso);
  const now = new Date();
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  );
};

const StatsCards: React.FC = () => {
  const [counts, setCounts] = React.useState({ rfp: 0, rfi: 0, rfq: 0 });
  const [loading, setLoading] = React.useState(false);

  const loadCounts = async () => {
    try {
      setLoading(true);
      const res = await apiClient.getFetchedEmails();
      let rfp = 0, rfi = 0, rfq = 0;
      const emailsPerAcct = Array.isArray(res?.emails) ? res.emails : [];
      for (const acct of emailsPerAcct) {
        const list = Array.isArray(acct.emails) ? acct.emails : [];
        for (const em of list) {
          if (!isToday(em.date)) continue;

          const subject = (em.subject || '').toLowerCase();
          const body    = (em.body_text || '').toLowerCase();              // NEW
          const attText = Array.isArray(em.attachment_names)                // NEW
            ? em.attachment_names.join(' ').toLowerCase()
            : ((em.attachments || []).map((a:any) =>
                (a.filename || a.file_path || a.url || '')
              ).join(' ').toLowerCase());

          const haystack = `${subject}\n${body}\n${attText}`;

          const isRFP = /\brfp\b|request\s+for\s+proposal/.test(haystack);
          const isRFI = /\brfi\b|request\s+for\s+information/.test(haystack);
          const isRFQ = /\brfq\b|request\s+for\s+quotation/.test(haystack);

          if (isRFP) rfp++;
          else if (isRFI) rfi++;
          else if (isRFQ) rfq++;
        }

      }
      setCounts({ rfp, rfi, rfq });
    } catch (e) {
      console.error('[StatsCards] failed to load counts', e);
      setCounts({ rfp: 0, rfi: 0, rfq: 0 });
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    loadCounts();
  }, []);

  const stats: StatItem[] = [
    {
      title: 'RFPs Today',
      count: counts.rfp,
      subtitle: loading ? 'Loading…' : 'Received today',
      color: '#3b82f6',
      icon: '📄'
    },
    {
      title: 'RFIs Today',
      count: counts.rfi,
      subtitle: loading ? 'Loading…' : 'Received today',
      color: '#8b5cf6',
      icon: '✉️'
    },
    {
      title: 'RFQs Today',
      count: counts.rfq,
      subtitle: loading ? 'Loading…' : 'Received today',
      color: '#f59e0b',
      icon: '📋'
    },
    // {
    //   title: 'No-Go Recommendations',
    //   count: 4,
    //   subtitle: 'Declined',
    //   color: '#ef4444',
    //   icon: '❌'
    // }
  ];

  return (
    <div className="mb-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Daily Stats</h2>
        <button
          onClick={loadCounts}
          disabled={loading}
          className={`text-sm px-4 py-1.5 rounded-lg font-medium shadow-sm transition-colors
            ${loading
              ? 'bg-blue-300 text-white cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white'}
          `}
        >
          {loading ? 'Refreshing…' : '🔄 Refresh'}
        </button>

      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, index) => (
          <div
            key={index}
            className={`rounded-xl shadow-lg border p-6 hover:shadow-xl transition-shadow duration-200 ${
              stat.isPreview 
                ? 'bg-gradient-to-br from-amber-50 to-orange-50 border-amber-200' 
                : 'bg-white border-gray-100'
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-2xl">{stat.icon}</span>
              <div className="text-right">
                <h3 className="text-sm font-medium text-gray-600">
                  {stat.title}
                </h3>
                {stat.isPreview && (
                  <div className="text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded-full mt-1">
                    🔧 Preview
                  </div>
                )}
              </div>
            </div>
            <div className="text-3xl font-bold mb-2" style={{ color: stat.color }}>
              {stat.count}
            </div>
            <div className={`text-sm ${stat.isPreview ? 'text-amber-600 font-medium' : 'text-gray-500'}`}>
              {stat.subtitle}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StatsCards;
