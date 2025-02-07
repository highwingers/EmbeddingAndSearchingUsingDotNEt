// See https://aka.ms/new-console-template for more information
using System.Diagnostics;

// Place txt files under python/data folder
// Run python/process-embed_vectorstore (once for new content)
// Run the Console app


// *** concept **/
// Chunk Large document into small data sentances, called chunking (ex, 500 tokens) .
// Apply Embedding to Chunks (embedding is storing as numbers)
// Store Embeding into Vector DB
// Query the Vector DB with your search term
// now pass the result of search from venctor DB to your fav ollama model (set it as context)

//string query = "find google ip?"; // Example query
var query = Console.ReadLine();
// Edit the path of file
string result = RunPythonScript("C:\\Users\\skhan\\Desktop\\Chunks_Embedings_Search\\Python\\search_with_mistral.py", query);

Console.WriteLine(result);
static string RunPythonScript(string scriptPath, string query)
{
    try
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = "python",  // Ensure "python" is in your system PATH
            Arguments = $"{scriptPath} \"{query}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (Process process = new Process { StartInfo = psi })
        {
            process.Start();

            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();

            process.WaitForExit();

            if (!string.IsNullOrEmpty(error))
            {
                Console.WriteLine($"Python Error: {error}");
                return null;
            }

            return output.Trim();
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error running Python script: {ex.Message}");
        return null;
    }
}