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
    <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 mb-2">Recent Documents</h2>
        <p className="text-gray-600">Latest RFP, RFI, and RFQ documents with their current status</p>
      </div>
 
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 font-medium text-gray-700">Type</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">File Name</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Status</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Assigned To</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc, index) => (
              <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                <td className="py-4 px-4">
                  <span className="flex items-center space-x-2">
                    <span>📄</span>
                    <span className="font-medium text-gray-900">{doc.type}</span>
                  </span>
                </td>
                <td className="py-4 px-4 text-gray-700">{doc.fileName}</td>
                <td className="py-4 px-4">{getStatusBadge(doc.status)}</td>
                <td className="py-4 px-4 text-gray-700">{doc.assignedTo}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
 
export default RecentDocuments;