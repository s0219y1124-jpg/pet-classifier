'use client'

import { useState } from 'react'
import Image from 'next/image'

export default function Home() {

  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleUpload = async () => {

    if (!file) {
      alert('画像を選択してください')
      return
    }

    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/predict`,
        {
          method: 'POST',
          body: formData,
        }
      )

      const data = await response.json()

      setResult(data)

    } catch (error) {

      console.error(error)
      alert('エラーが発生しました')

    }

    setLoading(false)
  }

  return (

    <>

    <Image
      src="/images/header.jpg"
      alt="header"
      width={1980}
      height={400}
      className="
        w-full
        h-70
        object-cover
        shadow-lg
        mb-8
      "
      />
    
    <main className="
    flex
    flex-col
    items-center
    bg-gradient-to-b
    from-white
    to-gray-100
    px-10
    py-16
    ">

      <h1 className="text-5xl font-bold mb-10">
        ペットAI判定
      </h1>

      <p className="
      text-gray-600
      mb-6
      text-center
      ">
      このサイトではペットとしてメジャーである犬、猫、ウサギ、ネズミの判別を行います。
      </p>

      <div className="
      flex
      gap-6
      mb-8
      ">
         <div className="text-center">
          <div className="text-4xl">🐶</div>
          <p>犬</p>
         </div>

         <div className="text-center">
          <div className="text-4xl">🐱</div>
          <p>猫</p>
         </div>

         <div className="text-center">
          <div className="text-4xl">🐰</div>
          <p>うさぎ</p>
         </div>

         <div className="text-center">
          <div className="text-4xl">🐭</div>
          <p>ねずみ</p>
         </div>

      </div>

      <label
      className="
      bg-blue-500
      hover:bg-blue-600
      text-white
      px-6
      py-3
      rounded-xl
      cursor-pointer
      inline-block
      "
      >
      画像を選択
      <input
      type="file"
      accept="image/*"
      className="hidden"
      onChange={(e) => {
        if (e.target.files) {
          setFile(e.target.files[0])
        }
      }}
      />
      </label>

      {file && (
        <img 
        src={URL.createObjectURL(file)}
        alt="preview"
        className="w-64 mt-5 rounded"
        />
      )}

      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded mt-5"
      >
        判定
      </button>

      {loading && (
        <p className="mt-5">
          判定中...
        </p>
      )}

      {result && (
        <div className="mt-5 border p-5 rounded">

          <p className="text-3xl font-bold text-blue-600">
            判定結果:
            <strong>
              {result.prediction}
            </strong>
          </p>

          <p>
            推論時間:
            {result.inference_time} 秒
          </p>

          <h2 className="mt-3 font-bold">
            スコア
          </h2>

          <div className="mt-5 w-80">
            {Object.entries(result.scores).map(
              ([label, score]) => {
                const percent = Number(score) * 100
                return (
                <div
                key={label}
                className="mb-4"
                >

                <div className="
                flex
                justify-between
                mb-1
                ">
                <span className="font-medium">
                  {label}
                </span>

                <span>
                  {percent.toFixed(1)}%
               </span>
              </div>

              <div className="
                w-full
                bg-gray-300
                rounded-full
                h-5
              ">
                <div
                className="
                bg-blue-500
                h-5
                rounded-full
                transition-all
                duration-500
                "
                  style={{
                    width: `${percent}%`
                  }}
                />

          </div>

        </div>
      )
    }
  )}

</div>

        </div>
      )}

    </main>

    </>
  )
}

