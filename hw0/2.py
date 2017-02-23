import Image
import sys
image_origin = Image.open(sys.argv[1])
image_compare = Image.open(sys.argv[2])
im=Image.new("RGBA",(image_compare.size[0],image_compare.size[1]),"white")

for x in range(image_compare.size[0]):
	for y in range(image_compare.size[1]):
		if image_compare.getpixel((x,y)) != image_origin.getpixel((x,y)):
			im.putpixel((x,y), image_compare.getpixel((x,y)))
		else:
			im.putpixel((x,y), (0,0,0,0))
im.save("ans_two.png")