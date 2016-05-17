

import java.io.DataInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class SpeechServer {
	private final static String path = "files/";
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			ServerSocket ss = new ServerSocket(9999);
			while (true){
				Socket cs = ss.accept();
				System.out.println(cs.getInetAddress().getHostAddress());
				DataInputStream in = new DataInputStream(cs.getInputStream());
				String data = in.readUTF();
				String[] split = data.split("\n\n");
				String headers = split[0];
				String body = split[1];
				String filename = headers.split("\n")[0];
				int size = Integer.parseInt(headers.split("\n")[1]);
				
				FileWriter f = new FileWriter(path + filename);
				f.write(body);
				f.close();
				
				cs.close();
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
