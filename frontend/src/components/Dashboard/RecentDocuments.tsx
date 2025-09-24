import React from 'react';
 
interface Document {
  type: string;
  fileName: string;
  status: string;
  assignedTo: string;
}
 
const RecentDocuments: React.FC = () => {
  const documents: Document[] = [
    {
      type: 'RFQ',
      fileName: 'Document_001_RFQ.pdf',
      status: 'In Progress',
      assignedTo: 'Jane Smith'
    },
    {
      type: 'RFQ',
      fileName: 'Document_002_RFQ.pdf',
      status: 'Completed',
      assignedTo: 'Mike Johnson'
    }
  ];
 
  const getStatusBadge = (status: string) => {
    const baseClasses = "px-3 py-1 rounded-full text-xs font-medium";
    const statusClasses = status === 'Completed'
      ? 'bg-green-100 text-green-800'
      : 'bg-yellow-100 text-yellow-800';
   
    return (
      <span className={`${baseClasses} ${statusClasses}`}>
        {status}
      </span>
    );
  };
 
  return (
    <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl shadow-lg border border-amber-200 p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-bold text-gray-900">Recent Documents</h2>
          <div className="text-xs bg-amber-100 text-amber-700 px-3 py-1 rounded-full font-medium">
            🔧 Preview Mode
          </div>
        </div>
        <p className="text-amber-700 font-medium">Latest RFP, RFI, and RFQ documents with their current status</p>
        <p className="text-sm text-amber-600 mt-1">Assignment feature is still in development</p>
      </div>
 
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-amber-200">
              <th className="text-left py-3 px-4 font-medium text-amber-800">Type</th>
              <th className="text-left py-3 px-4 font-medium text-amber-800">File Name</th>
              <th className="text-left py-3 px-4 font-medium text-amber-800">Status</th>
              <th className="text-left py-3 px-4 font-medium text-amber-800">Assigned To</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc, index) => (
              <tr key={index} className="border-b border-amber-100 hover:bg-amber-50 transition-colors">
                <td className="py-4 px-4">
                  <span className="flex items-center space-x-2">
                    <span>📄</span>
                    <span className="font-medium text-gray-900">{doc.type}</span>
                  </span>
                </td>
                <td className="py-4 px-4 text-gray-700">{doc.fileName}</td>
                <td className="py-4 px-4">{getStatusBadge(doc.status)}</td>
                <td className="py-4 px-4 text-gray-700">
                  <span className="flex items-center space-x-2">
                    <span>{doc.assignedTo}</span>
                    <span className="text-xs bg-amber-100 text-amber-600 px-2 py-0.5 rounded-full">
                      Preview
                    </span>
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
 
export default RecentDocuments;