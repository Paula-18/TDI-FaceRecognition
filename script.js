const imageUpload = document.getElementById('imageUpload');

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
    faceapi.nets.ssdModilenetv1.loadFromUri('./models')
]).then(start)

async function start(){
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)
    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    let image
    let canvas
    document.body.append('Loaded')

    imageUpload.addEventListener('change', async ()=> {
        if(image) image.remove()
        if(canvas) canvas.remove()

        image = await faceapi.bufferToImage(imageUpload.file[0])
        container.append(image)
        canvas.createCanvasFromMedia(canvas)
        container.append(canvas)

        const displaySize = { width: image.width, height: image.height }
        faceapi.matchDimensions(canvas, displaySize)

        const detections = await faceapi.detectAllFaces(image).withFaceLandMarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map(d => FaceMatcher.findBestMatch(d.descriptor))

        results.array.forEach((results, i) => {

            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.drawBox(box, {label: results.toString() })
            drawBox.draw(canvas)
            
        })
    })
}

function loadLabeledImages() {
    const labels = ['Black Widow', 'Captain America', 'Captian Marvel', 'Hawkeye', 'Jime Rhodes', 'Thor', 'Thony Stark']
    return Promise.all(
        label.map(async label => {
            const descriptions = []
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImagen(`https://mawe.mx/face/images/${label}/${i}.jpg`)
                const detections = await faceapi.detectionSingleFace(img).withFaceLandMarks().withFaceDescriptors()
                descriptions.push(detections.descriptor)
            }

        return new faceapi.labeledFaceDescriptors(label, descriptions)
        })
    )
}