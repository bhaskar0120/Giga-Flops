import std.stdio: writeln;
import std.file:read;
import core.time;

void main(){
  writeln("Hello world");
  immutable N = 1024;
  immutable tmp = cast(immutable(void[]))read("mat.dat", 2*N*N*float.sizeof);
  immutable float[] a = cast(float[])tmp[0..N*N*float.sizeof];
  immutable float[] b = cast(float[])tmp[N*N*float.sizeof..$];
  float[] c = new float[N*N];
  foreach(ref i ; c)
    i = 0;

  auto st = MonoTime.currTime();
  for(int k = 0; k < N; ++k){
    for(int i = 0; i < N; ++i){
      for(int j = 0; j < N; ++j){
        c[i*N+j] += a[i*N+k]*a[k*N+j];
      }
    }
  }
  auto et = MonoTime.currTime();
  long nanos;
  (et-st).split!("nsecs")(nanos);
  float flop = 2.0*N*N*N;
  float gflops = flop/nanos;
  for(int i = 0; i < N*N; ++i)
    if(c[i] != b[i]){
      writeln("Fail, Mismatch at i, ",c[i]," != ",b[i]);
      return ;
    }
  writeln(gflops);

}


