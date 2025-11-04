รายงานโครงงาน วิชา Deep learning  
หัวข้อ :Ai ทำให้ภาพ ชัดขึ้น   
ที่มาและความสำคัญ:   
เนื่องจากโทรศัพท์ของผม มีคุณภาพของกล้องที่ไม่ได้มีคุณภาพมากนัก เลยทำให้เวลาที่เราถ่ายรูปหรือ พยายาม zoom รูป ให้ไกลๆ ก็จะได้รูปที่ ไม่ค่อยชัดหรือ มีคุณภาพต่ำ ซึ่งทำให้ผมได้เกิด ความคิดที่ว่า เราจะสามารถมารถใช้ model deep learning ในการทำให้ภาพชัดได้หรือไม่ โดนการ ที่เรา ที่ เรามี input เป็นรูปที่ ไม่ชัดจากนั้น ให้ model ได้ไป predict ออกมาว่าเป็นอย่างไร โดยเทียบกับ รูปที่ชัด  
วิธีการ: ใช้ model SRGAN ในการทำนายภาพออกมาโดย ให้ ออกแบบแต่ละ layer ดังนี้  
Discriminator:  
<img width="110" height="567" alt="ภาพถ่ายหน้าจอ 2568-11-04 เวลา 23 08 15" src="https://github.com/user-attachments/assets/14bfe2c1-d9f0-4b6a-8e6e-59ae5ebf71bf" />


Generator:
<img width="106" height="447" alt="ภาพถ่ายหน้าจอ 2568-11-04 เวลา 23 09 46" src="https://github.com/user-attachments/assets/53cedd43-93ef-428b-b208-ea2e18347e6c" />
  

โดย แต่ละส่วนของ code มีดังนี้ 
1. main.py ซึ่งเป็น โคดที่เอาไว้ใช้ train model ซึ่งเทรนใน Kaggle ใช้ GPU Nvidia T4 / ตัว ในการเทรน  ตัวแรกเทรน Generator อีกตัวเทรนให้ discriminator  
2. dataset.py เป็น codeที่ preprocress dataset ด้วยการ อ่าน dataset จาก DIV2K จากนั้นก็เอา ไปให้ไฟล์ main.py เทรนต่อไป  
3. model.py เป็น ส่วนที่สร้างตัว model ขึ้นมาซึ่งได้อธิบายไปก่อนหน้าแล้ว   
4. vgg\_loss.py เป็นส่วนที่ เอา model vgg มาช่วย คำนวณค่า loss  
5. inference.py เป็นส่วนที่ลอง เอา model มาใช้จริง

อธิบายวิธีในการ train  
Train discriminator:  
	ให้ Generator ทำให้ ภาพที่รับมา ชัดขึ้น (โดยไม่คำนวณ back propergation) จากนั้น ก็ให้ discriminator ทาย จากนั้นทำซ้ำกับ label รูปที่ ชัดจากนั้น เอา ค่าที่ได้ทั้งคู่ไปคำนวณค่า loss หารเฉลี่ยออกมา จากนั้น หาร ด้วย gradient accumulation step เนื่องจาก ข้อมูลภาพที่ได้มีหลายขนาดเลยทำให้ ไม่สามารถเอามา stack กันได้ จากนั้น ก็ปรับค่า parameters ของ discriminator เมื่อ ทำ  
back propergation ครบ 16 ครั้ง (ตาม gradient accumulation step)  
Train generator:  
	ให้ generator ทำให้ภาพที่รับมาชัดขึ้น จากนั้น เอาไปให้ discriminator ทายค่าแล้วคำนวณ loss อันแรก จากนั้น ก็เอาไปเทียบ กับ ภาพชัดจริงๆ ทีละ pixel ด้วย mean square error เป็น loss ตัวที่ 2 จากนั้น เอา model vgg มาช่วย คำนวณ l1 loss เป็น loss ตัวที่ 3  
เอา ทั้ง 3 มารวมกัน ตามสมการ   
<img width="520" height="109" alt="ภาพถ่ายหน้าจอ 2568-11-04 เวลา 23 22 08" src="https://github.com/user-attachments/assets/d83d0a55-d1b7-4172-b8af-642f7904ded0" />
แล้วก็
ปรับค่า weight เหมือนกับ Discriminator  
อ้างอิง:  
	Model: [https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802)  
Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
