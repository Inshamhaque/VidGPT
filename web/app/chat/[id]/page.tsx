interface PageProps {
    params : {
        id : string
    }
}
export default function ChatPage({ params }: PageProps) {
  const { id } = params;
  
  return (
    <div className="grid grid-cols-2 h-screen">
    {/* chat side */}
      <div className="flex flex-col h-screen col-span-1 bg-gray-600">
        {/* message container */}
        <div className="flex-1 overflow-y-auto"></div>
        {/* input box */}
        <div className="bottom-0 bg-gray-100">
            <textarea className="bottom-0" />
        </div>
      </div>
    {/* video player side */}
      <div className="col-span-1 bg-gray-300">

      </div>
    </div>
  );
}
