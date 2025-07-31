import { Home } from "lucide-react"
import { useRouter } from "next/navigation"

export default function Navbar(){
    const router = useRouter()
  return(
    <div className="w-full flex justify-between items-center p-4 bg-[#1a1a1a] border-b border-gray-700">
      <div className="text-xl font-bold text-white">VIDEOGPT</div>
      <div onClick={()=>{
        router.push('/')
      }} className="flex items-center gap-2 text-white hover:text-blue-400 cursor-pointer transition-colors">
        <Home size={18} />
        <span>Home</span>
      </div>
    </div>
  )
}