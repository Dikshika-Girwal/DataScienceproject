result = 42
text_output = "This is a simple output."

# Save output to a file
with open('output.txt', 'w') as f:
    f.write("Result: {}\n".format(result))
    f.write("Text Output: {}\n".format(text_output))
 
print("Result:", result)
print("Text Output:", text_output)
 