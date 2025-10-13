
const JsonDisplay = ({ data }) => {
  const formattedJson = JSON.stringify(data, null, 2);

  return (
    <div className="bg-gray-50 p-4 rounded-lg overflow-auto h-full">
      <pre className="text-sm font-mono whitespace-pre-wrap text-gray-800">
        {formattedJson}
      </pre>
    </div>
  );
};

export default JsonDisplay;