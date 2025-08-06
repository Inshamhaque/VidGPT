'use client'

import Image from 'next/image'
import { Home, Sun } from 'lucide-react'
import { useRouter } from 'next/navigation'
import logo from '../../public/Screenshot_2025-08-05_at_4.56.14_PM-removebg-preview.png'

export default function Navbar() {
  const router = useRouter()

  return (
    <div className="w-full flex justify-between items-center p-4 bg-black/80 border-b border-gray-700">
      <Image onClick={()=>{
        router.push('/')
      }} src={logo} alt="App Logo" width={120} height={40}  />
      <button className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-all duration-300">
          <Sun />
      </button>

    </div>
  )
}
